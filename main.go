package main

import (
	"context"
	"os"

	"lcbchat/chunking"
	"lcbchat/scraper"
	"lcbchat/embedding"
	"lcbchat/vectordb"
	"log"
	"time"
)

func main() {
	// Initialize output dir
	outputFolder := "./scraped_data"
	if err := os.MkdirAll(outputFolder, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
		return
	}
	
	// Initialize the scraper configuration
	scrapeConfig := scraper.ScraperConfig{
		StartURL:        "https://www.oregon.gov/lcb/Pages/Index.aspx",
		DomainPrefix:    "https://www.oregon.gov/lcb/",
		MaxDepth:        40, // For LCB: 40 is a good dept
		MaxConcurrency:  5,
		DelayBetween:    1 * time.Second, // Reduced delay
		RespectRobots:   false, // Temporarily disable for testing
		OutDir: 	     outputFolder,
		UserAgent:       "RAG-Scraper/1.0",
		EnableDebug:     true, // Enable debug
	}

	scraper, err := scraper.NewWebScraper(scrapeConfig)
	if err != nil {
		log.Fatalf("Failed to create scraper: %v", err)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Start scraping
	if err := scraper.Start(ctx); err != nil {
		log.Fatalf("Scraping failed: %v", err)
	}

	content := scraper.GetContent()
	log.Printf("Scraping completed. Found %d unique content items", len(content))

	// Print summary
	htmlCount := 0
	pdfCount := 0
	for _, item := range content {
		switch item.ContentType {
		case "text/html":
			htmlCount++
		case "application/pdf":
			pdfCount++
		}
	}
	
	log.Printf("Content breakdown: %d HTML pages, %d PDF documents", htmlCount, pdfCount)

	// Process and chunk the scraped content
	chunkConfig := chunking.DefaultChunkingConfig()
	if err := chunking.ProcessAndSave(chunkConfig, scraper.GetContent(), outputFolder); err != nil {
		log.Printf("Failed to process and save chunks: %v", err)
		return
	}

	// Embed the scraped content
	embedConfig := embedding.DefaultEmbeddingConfig()
	embedConfig.Model = "nomic-embed-text"

	ollamaUrl := "http://localhost:11434"
	chunksFile := outputFolder + "/chunks_all.json"
	outputDir := "./output"

	//Process chunks to embedding
	if err := embedding.ProcessChunksToEmbeddings(chunksFile, outputDir, ollamaUrl, embedConfig); err != nil {
		log.Fatalf("Failed to process chunks to embeddings: %v", err)
	}

	// Write output data to qdrant database
	vdbConfig := vectordb.DefaultVectorDBConfig()
	vdbConfig.CollectionName = "lcbchat"
	vdbConfig.VectorSize = 768 // 768 for nomic-embed-text

	embeddedChunksFile := outputDir + "/embedded_chunks_all.json"
	vdb, err := vectordb.SetupVectorDatabase(embeddedChunksFile, vdbConfig)
	if err != nil {
		log.Fatalf("Failed to setup vector database: %v", err)
	}
	defer vdb.Close()

	// SAMPLE SEARCH:
	// Normally you would have a query vector from an embedding model.
	queryVector := make([]float64, 768) // Dummy query vector
	
	results, err := vdb.Search(ctx, queryVector, 5, nil)
	if err != nil {
		log.Printf("Search error: %v", err)
		return
	}

	log.Printf("Found %d results", len(results))
	for i, result := range results {
		log.Printf("Result %d: Score=%.4f, Title=%s", 
			i+1, result.Score, result.Chunk.Title)
	}
}