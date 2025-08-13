package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"lcbchat/chunking"
	"lcbchat/embedding"
	"lcbchat/rag"
	"lcbchat/scraper"
	"lcbchat/vectordb"
	"log"
	"log/slog"

	"time"
	"flag"
)

const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorWhite  = "\033[37m"
	ColorBold   = "\033[1m"
)

// Config holds all configuration for the application
type Config struct {
	RAGConfig      rag.RAGConfig
	OutputDir      string
	ScraperConfig  scraper.ScraperConfig
	ChunkingConfig chunking.ChunkingConfig
	EmbeddingConfig embedding.EmbeddingConfig
}

func main() {
	// Define command line flags
	var (
		command = flag.String("cmd", "interactive", "Command to run: scrape, query, interactive")
		query   = flag.String("query", "", "Query to ask (for single query mode)")
		//url     = flag.String("url", "", "URL to scrape")
		verbose = flag.Bool("verbose", false, "Enable verbose logging")
		topK    = flag.Int("topk", 5, "Number of results to retrieve")
		model   = flag.String("model", "llama3.1:8b", "Generation model to use")
		help    = flag.Bool("help", false, "Show help")
	)
	flag.Parse()

	if *help {
		showHelp()
		return
	}

	// Setup logging
	if *verbose {
		logHandler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
		slog.SetDefault(slog.New(logHandler))
	}

	// Initialize configuration
	config, err := initConfigs()
	if err != nil {
		log.Fatalf("Failed to initialize configurations: %v", err)
		os.Exit(1)
	}

	// Override config with command line args
	config.RAGConfig.TopK = *topK
	config.RAGConfig.GenerationModel = *model

	// Execute the requested command
	switch *command {
	case "scrape":
		runScrapeAndIndex(config)
	case "query":
		if *query == "" {
			fmt.Printf("%sError: --query is required for single query mode%s\n", ColorRed, ColorReset)
			os.Exit(1)
		}
		runSingleQuery(*query, config)
	case "interactive":
		runInteractiveMode(config)
	default:
		fmt.Printf("%sError: Unknown command '%s'%s\n", ColorRed, *command, ColorReset)
		showHelp()
		os.Exit(1)
	}
}

func showHelp() {
	fmt.Printf("%s%sRAG System CLI%s%s\n\n", ColorBold, ColorBlue, ColorReset, ColorReset)
	fmt.Println("Usage:")
	fmt.Printf("  %s./program --cmd=<command> [options]%s\n\n", ColorCyan, ColorReset)
	
	fmt.Printf("%sCommands:%s\n", ColorBold, ColorReset)
	fmt.Println("  scrape       - Scrape a website and process into vector database")
	fmt.Println("  query        - Ask a single question")
	fmt.Println("  interactive  - Start interactive chat mode (default)")
	
	fmt.Printf("\n%sOptions:%s\n", ColorBold, ColorReset)
	fmt.Println("  --url        URL to scrape")
	fmt.Println("  --query      Question to ask")
	fmt.Println("  --topk       Number of results to retrieve (default: 5)")
	fmt.Println("  --model      Generation model to use (default: llama3.1:8b)")
	fmt.Println("  --verbose    Enable verbose logging")
	fmt.Println("  --help       Show this help")
	
	fmt.Printf("\n%sExamples:%s\n", ColorBold, ColorReset)
	fmt.Printf("  %s./program --cmd=interactive%s\n", ColorGreen, ColorReset)
	fmt.Printf("  %s./program --cmd=query --query=\"How do I reset my password?\"%s\n", ColorGreen, ColorReset)
}

func runSingleQuery(query string, config Config) {
	fmt.Printf("%s🤖 Initializing RAG system...%s\n", ColorBlue, ColorReset)
	
	ragSystem, err := rag.NewRAGSystem(config.RAGConfig)
	if err != nil {
		log.Fatalf("Failed to initialize RAG system: %v", err)
	}
	defer ragSystem.Close()

	fmt.Printf("%s❓ Querying: %s%s\n", ColorCyan, query, ColorReset)
	
	ctx := context.Background()
	result, err := ragSystem.Query(ctx, query)
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}

	printQueryResult(result)
}

func runInteractiveMode(config Config) {
	fmt.Printf("%s%s🚀 Starting Interactive RAG Chat%s%s\n", ColorBold, ColorGreen, ColorReset, ColorReset)
	fmt.Printf("%sInitializing system...%s\n", ColorYellow, ColorReset)
	
	ragSystem, err := rag.NewRAGSystem(config.RAGConfig)
	if err != nil {
		log.Fatalf("Failed to initialize RAG system: %v", err)
	}
	defer ragSystem.Close()

	fmt.Printf("%s✅ System ready!%s\n", ColorGreen, ColorReset)
	fmt.Printf("Using model: %s%s%s\n", ColorCyan, config.RAGConfig.GenerationModel, ColorReset)
	fmt.Printf("Vector database: %s%s%s\n", ColorCyan, config.RAGConfig.VectorDBConfig.CollectionName, ColorReset)
	fmt.Println()
	printInteractiveHelp()

	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()
	
	for {
		fmt.Printf("%s> %s", ColorBlue, ColorReset)
		
		if !scanner.Scan() {
			break
		}
		
		input := strings.TrimSpace(scanner.Text())
		
		if input == "" {
			continue
		}
		
		// Handle special commands
		if handleSpecialCommand(input, ragSystem, &config) {
			continue
		}
		
		// Process the query
		fmt.Printf("%s🔍 Searching...%s\n", ColorYellow, ColorReset)
		result, err := ragSystem.Query(ctx, input)
		if err != nil {
			fmt.Printf("%sError: %v%s\n", ColorRed, err, ColorReset)
			continue
		}
		
		printQueryResult(result)
		fmt.Println()
	}
	
	fmt.Printf("\n%s👋 Goodbye!%s\n", ColorGreen, ColorReset)
}

func printInteractiveHelp() {
	fmt.Printf("%sCommands:%s\n", ColorBold, ColorReset)
	fmt.Printf("  %s/help%s      - Show this help\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/stats%s     - Show system statistics\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/config%s    - Show current configuration\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/topk <n>%s  - Change number of results retrieved\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/model <m>%s - Change generation model\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/similar%s   - Show similar chunks without generating answer\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/quit%s      - Exit the program\n", ColorCyan, ColorReset)
	fmt.Println()
}

func handleSpecialCommand(input string, ragSystem *rag.RAGSystem, config *Config) bool {
	switch {
	case input == "/quit" || input == "/exit":
		fmt.Printf("%s👋 Goodbye!%s\n", ColorGreen, ColorReset)
		os.Exit(0)
		
	case input == "/help":
		printInteractiveHelp()
		return true
		
	case input == "/stats":
		printSystemStats(config)
		return true
		
	case input == "/config":
		printCurrentConfig(config)
		return true
		
	case strings.HasPrefix(input, "/topk "):
		parts := strings.Fields(input)
		if len(parts) != 2 {
			fmt.Printf("%sUsage: /topk <number>%s\n", ColorRed, ColorReset)
			return true
		}
		
		n, err := strconv.Atoi(parts[1])
		if err != nil || n < 1 || n > 20 {
			fmt.Printf("%sError: topk must be a number between 1 and 20%s\n", ColorRed, ColorReset)
			return true
		}
		
		config.RAGConfig.TopK = n
		ragSystem.UpdateConfig(config.RAGConfig)
		fmt.Printf("%s✅ TopK set to %d%s\n", ColorGreen, n, ColorReset)
		return true
		
	case strings.HasPrefix(input, "/model "):
		parts := strings.Fields(input)
		if len(parts) != 2 {
			fmt.Printf("%sUsage: /model <model_name>%s\n", ColorRed, ColorReset)
			return true
		}
		
		config.RAGConfig.GenerationModel = parts[1]
		fmt.Printf("%s⚠️  Model changed to %s (restart required for full effect)%s\n", 
			ColorYellow, parts[1], ColorReset)
		return true
		
	case strings.HasPrefix(input, "/similar"):
		query := strings.TrimPrefix(input, "/similar")
		query = strings.TrimSpace(query)
		
		if query == "" {
			fmt.Printf("%sUsage: /similar <your query>%s\n", ColorRed, ColorReset)
			return true
		}
		
		ctx := context.Background()
		results, err := ragSystem.GetSimilarChunks(ctx, query, config.RAGConfig.TopK)
		if err != nil {
			fmt.Printf("%sError: %v%s\n", ColorRed, err, ColorReset)
			return true
		}
		
		printSimilarChunks(results)
		return true
	}
	
	return false
}

func printQueryResult(result *rag.QueryResult) {
	fmt.Printf("\n%s%s💡 Answer:%s%s\n", ColorBold, ColorGreen, ColorReset, ColorReset)
	fmt.Printf("%s\n", wrapText(result.Answer, 80))
	
	if len(result.Sources) > 0 {
		fmt.Printf("\n%s📚 Sources (%d):%s\n", ColorPurple, len(result.Sources), ColorReset)
		for i, source := range result.Sources {
			fmt.Printf("%s  %d. [Score: %.3f] %s%s\n", 
				ColorCyan, i+1, source.Score, source.Chunk.Title, ColorReset)
			fmt.Printf("     %s%s%s\n", ColorWhite, source.Chunk.SourceURL, ColorReset)
		}
	}
	
	fmt.Printf("\n%s⏱️  Timing:%s Retrieval: %v | Generation: %v | Total: %v\n", 
		ColorYellow, ColorReset, result.RetrievalTime, result.GenerationTime, result.ProcessingTime)
	
	if result.TokensGenerated > 0 {
		fmt.Printf("%s🔢 Tokens generated: %d%s\n", ColorYellow, result.TokensGenerated, ColorReset)
	}
}

func printSimilarChunks(results []vectordb.SearchResult) {
	fmt.Printf("\n%s📋 Similar Chunks:%s\n", ColorPurple, ColorReset)
	
	for i, result := range results {
		fmt.Printf("\n%s%d. [Score: %.3f] %s%s\n", 
			ColorCyan, i+1, result.Score, result.Chunk.Title, ColorReset)
		fmt.Printf("   %sURL:%s %s\n", ColorWhite, ColorReset, result.Chunk.SourceURL)
		fmt.Printf("   %sContent:%s %s\n", ColorWhite, ColorReset, 
			truncateText(result.Chunk.Content, 200))
	}
}

func printSystemStats(config *Config) {
	fmt.Printf("\n%s📊 System Statistics:%s\n", ColorPurple, ColorReset)
	fmt.Printf("Generation Model: %s%s%s\n", ColorCyan, config.RAGConfig.GenerationModel, ColorReset)
	fmt.Printf("Embedding Model:  %s%s%s\n", ColorCyan, config.RAGConfig.EmbeddingModel, ColorReset)
	fmt.Printf("Collection:       %s%s%s\n", ColorCyan, config.RAGConfig.VectorDBConfig.CollectionName, ColorReset)
	fmt.Printf("Top K:           %s%d%s\n", ColorCyan, config.RAGConfig.TopK, ColorReset)
	fmt.Printf("Score Threshold: %s%.2f%s\n", ColorCyan, config.RAGConfig.ScoreThreshold, ColorReset)
	fmt.Printf("Temperature:     %s%.2f%s\n", ColorCyan, config.RAGConfig.Temperature, ColorReset)
}

func printCurrentConfig(config *Config) {
	fmt.Printf("\n%s⚙️  Current Configuration:%s\n", ColorPurple, ColorReset)
	fmt.Printf("Ollama URL:       %s\n", config.RAGConfig.OllamaURL)
	fmt.Printf("Generation Model: %s\n", config.RAGConfig.GenerationModel)
	fmt.Printf("Embedding Model:  %s\n", config.RAGConfig.EmbeddingModel)
	fmt.Printf("Top K:           %d\n", config.RAGConfig.TopK)
	fmt.Printf("Score Threshold: %.2f\n", config.RAGConfig.ScoreThreshold)
	fmt.Printf("Max Tokens:      %d\n", config.RAGConfig.MaxTokens)
	fmt.Printf("Temperature:     %.2f\n", config.RAGConfig.Temperature)
}

// Utility functions
func wrapText(text string, width int) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}
	
	var lines []string
	var currentLine strings.Builder
	
	for _, word := range words {
		if currentLine.Len() > 0 && currentLine.Len()+len(word)+1 > width {
			lines = append(lines, currentLine.String())
			currentLine.Reset()
		}
		
		if currentLine.Len() > 0 {
			currentLine.WriteString(" ")
		}
		currentLine.WriteString(word)
	}
	
	if currentLine.Len() > 0 {
		lines = append(lines, currentLine.String())
	}
	
	return strings.Join(lines, "\n")
}

func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	
	truncated := text[:maxLen-3]
	lastSpace := strings.LastIndex(truncated, " ")
	if lastSpace > maxLen/2 {
		truncated = truncated[:lastSpace]
	}
	
	return truncated + "..."
}

func runScrapeAndIndex(config Config) {
	fmt.Printf("%s🚀 Running full pipeline for: %s%s\n", ColorBlue, config.ScraperConfig.DomainPrefix, ColorReset)

	// SCRAPE
	fmt.Printf("\n%s📥 Step 1: Scraping and indexing website...%s\n", ColorYellow, ColorReset)
	// Initialize scraper
	scraper, err := scraper.NewWebScraper(config.ScraperConfig)
	if err != nil {
		log.Fatalf("💥 Failed to create scraper: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Start scraping
	if err := scraper.Start(ctx); err != nil {
		log.Fatalf("Scraping failed: %v", err)
	}
	content := scraper.GetContent()
	fmt.Printf("%s✅ Scraping completed%s. Found %d unique content items", ColorGreen, ColorReset, len(content))

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
	
	fmt.Printf("Content breakdown: %d HTML pages, %d PDF documents", htmlCount, pdfCount)
	// Process and chunk the scraped content
	if err := chunking.ProcessAndSave(config.ChunkingConfig, scraper.GetContent(), config.ScraperConfig.OutDir); err != nil {
		log.Fatalf("%s💥 Failed to process and save chunks:%s %v", ColorRed, ColorReset, err)
	}

	// Embed the scraped content
	ollamaUrl := "http://localhost:11434"
	chunksFile := config.ScraperConfig.OutDir + "/chunks_all.json"

	//Process chunks to embedding
	fmt.Printf("%s🔗 Processing chunks to embeddings...%s\n", ColorYellow, ColorReset)
	if err := embedding.ProcessChunksToEmbeddings(chunksFile, config.OutputDir, ollamaUrl, config.EmbeddingConfig); err != nil {
		log.Fatalf("💥 Failed to process chunks to embeddings: %v", err)
	}
	fmt.Printf("%s✅ Embedding completed!%s\n", ColorGreen, ColorReset)

	// Write output data to qdrant database
	fmt.Printf("%s📦 Setting up vector database...%s\n", ColorYellow, ColorReset)
	embeddedChunksFile := config.OutputDir + "/embedded_chunks_all.json"
	vdb, err := vectordb.SetupVectorDatabase(embeddedChunksFile, vectordb.DefaultVectorDBConfig())
	if err != nil {
		log.Fatalf("Failed to setup vector database: %v", err)
	}
	defer vdb.Close()
	fmt.Printf("%s✅ Vector database setup complete!%s\n", ColorGreen, ColorReset)
	fmt.Printf("%s👍 Scraping and database work is completed! You may now procedd with asking questions.%s", ColorCyan, ColorReset)
}

func initConfigs() (Config, error) {
	// Initialize default configurations
	
	// Initialize the scraper configuration
	scrapeConfig := scraper.ScraperConfig{
		StartURL:        "https://www.oregon.gov/lcb/Pages/Index.aspx",
		DomainPrefix:    "https://www.oregon.gov/lcb/",
		MaxDepth:        40, // For LCB: 40 is a good dept
		MaxConcurrency:  5,
		DelayBetween:    1 * time.Second, // Reduced delay
		RespectRobots:   false, // Temporarily disable for testing
		OutDir: 	     "./scraped_data",
		UserAgent:       "RAG-Scraper/1.0",
		EnableDebug:     true, // Enable debug
	}

	config := Config{
		RAGConfig: rag.DefaultRAGConfig(),
		OutputDir: "./output",
		ScraperConfig: scrapeConfig,
		ChunkingConfig: chunking.DefaultChunkingConfig(),
		EmbeddingConfig: embedding.DefaultEmbeddingConfig(),
	}

	return config, nil
}
