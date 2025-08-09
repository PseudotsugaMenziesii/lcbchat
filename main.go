package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"lcbchat/chunking"
	"lcbchat/embedding"
	"lcbchat/scraper"
	"lcbchat/vectordb"
	"lcbchat/rag"
	"log"
	"time"
)


func main() {

	if len(os.Args) < 2 {
        fmt.Println("Usage: program <command>")
        fmt.Println("Commands: scrape, query, serve")
        os.Exit(1)
    }

    switch os.Args[1] {
    case "scrape":
        runScrapeAndIndex()
    case "query":
        runQuery(strings.Join(os.Args[2:], " "))
    case "serve":
        //runServer()
		fmt.Print("Server functionality is not implemented yet.\n")
    default:
        fmt.Printf("Unknown command: %s\n", os.Args[1])
    }

}

func runScrapeAndIndex() {
	fmt.Println("Running scrape and index...")
	
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
		embeddedChunksFile := outputDir + "/embedded_chunks_all.json"
		vdb, err := vectordb.SetupVectorDatabase(embeddedChunksFile, vectordb.DefaultVectorDBConfig())
		if err != nil {
			log.Fatalf("Failed to setup vector database: %v", err)
		}
		defer vdb.Close()
}

func runQuery(query string) {
	fmt.Printf("Running query: %s\n", query)
	
	config := rag.DefaultRAGConfig()
    config.GenerationModel = "llama3.1:8b"
    config.TopK = 3
    
    ragSystem, err := rag.NewRAGSystem(config)
    if err != nil {
        log.Fatal(err)
    }
    defer ragSystem.Close()

    // Query the system
    ctx := context.Background()
    result, err := ragSystem.Query(ctx, query)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    fmt.Printf("Question: %s\n", result.Query)
    fmt.Printf("Answer: %s\n", result.Answer)
    fmt.Printf("Sources: %d chunks\n", len(result.Sources))
    fmt.Printf("Processing time: %v\n", result.ProcessingTime)
}