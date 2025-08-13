package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"
	//"strconv"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/rs/cors"

	"lcbchat/chunking"
	"lcbchat/embedding"
	"lcbchat/rag"
	"lcbchat/scraper"
	"lcbchat/vectordb"
	"lcbchat/config"
)

// Server holds the HTTP server and RAG system
type Server struct {
	ragSystem    *rag.RAGSystem
	config       config.Config
	stats        *SystemStats
	isInitialized bool
	mu           sync.RWMutex
	
	// WebSocket upgrader for real-time updates
	upgrader websocket.Upgrader
}

// SystemStats tracks system performance
type SystemStats struct {
	QueryCount      int64     `json:"queryCount"`
	TotalTime       int64     `json:"totalTime"` // in milliseconds
	TokensGenerated int64     `json:"tokensGenerated"`
	SuccessCount    int64     `json:"successCount"`
	StartTime       time.Time `json:"startTime"`
	mu              sync.RWMutex
}

// API Request/Response types
type QueryRequest struct {
	Query string `json:"query"`
}

type QueryResponse struct {
	Answer          string                  `json:"answer"`
	Sources         []SourceInfo            `json:"sources"`
	ProcessingTime  int64                   `json:"processingTime"`
	RetrievalTime   int64                   `json:"retrievalTime"`
	GenerationTime  int64                   `json:"generationTime"`
	TokensGenerated int                     `json:"tokensGenerated"`
	Success         bool                    `json:"success"`
	Error           string                  `json:"error,omitempty"`
}

type SourceInfo struct {
	Title     string  `json:"title"`
	URL       string  `json:"url"`
	Score     float64 `json:"score"`
	Content   string  `json:"content,omitempty"`
}

type ConfigUpdateRequest struct {
	TopK        int     `json:"topk,omitempty"`
	Model       string  `json:"model,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type InitializeRequest struct {
	// Add any initialization parameters if needed
}

type InitializeResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}

type ScrapeRequest struct {
	URL string `json:"url,omitempty"` // Optional override
}

type ScrapeResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Error   string `json:"error,omitempty"`
}

type StatsResponse struct {
	*SystemStats
	AverageTime  int64  `json:"averageTime"`
	SuccessRate  float64 `json:"successRate"`
	CurrentModel string `json:"currentModel"`
	TopK         int    `json:"topK"`
	IsInitialized bool  `json:"isInitialized"`
}

type SimilarRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"topk,omitempty"`
}

type SimilarResponse struct {
	Results []SourceInfo `json:"results"`
	Success bool         `json:"success"`
	Error   string       `json:"error,omitempty"`
}

// WebSocket message types for real-time updates
type WSMessage struct {
	Type    string      `json:"type"`
	Data    interface{} `json:"data"`
	Message string      `json:"message,omitempty"`
}

func NewServer(config config.Config) *Server {

	return &Server{
		config: config,
		stats: &SystemStats{
			StartTime: time.Now(),
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins in development
			},
		},
	}
}

func (s *Server) setupRoutes() http.Handler {
	r := mux.NewRouter()

	// API routes
	api := r.PathPrefix("/api").Subrouter()
	api.HandleFunc("/initialize", s.handleInitialize).Methods("POST")
	api.HandleFunc("/query", s.handleQuery).Methods("POST")
	api.HandleFunc("/scrape", s.handleScrape).Methods("POST")
	api.HandleFunc("/stats", s.handleStats).Methods("GET")
	api.HandleFunc("/config", s.handleConfigUpdate).Methods("POST")
	api.HandleFunc("/similar", s.handleSimilar).Methods("POST")
	api.HandleFunc("/ws", s.handleWebSocket).Methods("GET")

	// Serve static files (your HTML interface)
	r.PathPrefix("/").Handler(http.FileServer(http.Dir("./static/")))

	// Setup CORS
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"*"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
	})

	return c.Handler(r)
}

func (s *Server) handleInitialize(w http.ResponseWriter, r *http.Request) {
	var req InitializeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.isInitialized {
		s.respondJSON(w, InitializeResponse{
			Success: true,
			Message: "System already initialized",
		})
		return
	}

	log.Println("Initializing RAG system...")

	ragSystem, err := rag.NewRAGSystem(s.config.RAGConfig)
	if err != nil {
		s.respondJSON(w, InitializeResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to initialize RAG system: %v", err),
		})
		return
	}

	s.ragSystem = ragSystem
	s.isInitialized = true

	log.Println("RAG system initialized successfully")

	s.respondJSON(w, InitializeResponse{
		Success: true,
		Message: "RAG system initialized successfully",
	})
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if !s.isInitialized {
		http.Error(w, "System not initialized", http.StatusPreconditionFailed)
		return
	}

	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "Query cannot be empty", http.StatusBadRequest)
		return
	}

	start := time.Now()
	ctx := context.Background()

	result, err := s.ragSystem.Query(ctx, req.Query)
	processingTime := time.Since(start).Milliseconds()

	response := QueryResponse{
		ProcessingTime: processingTime,
	}

	if err != nil {
		response.Success = false
		response.Error = err.Error()
	} else {
		response.Success = true
		response.Answer = result.Answer
		response.RetrievalTime = result.RetrievalTime.Milliseconds()
		response.GenerationTime = result.GenerationTime.Milliseconds()
		response.TokensGenerated = result.TokensGenerated

		// Convert sources
		for _, source := range result.Sources {
			response.Sources = append(response.Sources, SourceInfo{
				Title:   source.Chunk.Title,
				URL:     source.Chunk.SourceURL,
				Score:   float64(source.Score),
				Content: source.Chunk.Content[:min(200, len(source.Chunk.Content))], // Truncate for API
			})
		}

		// Update stats
		s.updateStats(processingTime, result.TokensGenerated, true)
	}

	if !response.Success {
		s.updateStats(processingTime, 0, false)
	}

	s.respondJSON(w, response)
}

func (s *Server) handleScrape(w http.ResponseWriter, r *http.Request) {
	var req ScrapeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Use configured URL if not provided
	if req.URL == "" {
		req.URL = s.config.ScraperConfig.StartURL
	}

	go s.runScrapeProcess(req.URL) // Run in background

	s.respondJSON(w, ScrapeResponse{
		Success: true,
		Message: "Scraping started in background",
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()

	var avgTime int64
	var successRate float64

	if s.stats.QueryCount > 0 {
		avgTime = s.stats.TotalTime / s.stats.QueryCount
		successRate = float64(s.stats.SuccessCount) / float64(s.stats.QueryCount) * 100
	} else {
		successRate = 100
	}

	response := StatsResponse{
		SystemStats:   s.stats,
		AverageTime:   avgTime,
		SuccessRate:   successRate,
		CurrentModel:  s.config.RAGConfig.GenerationModel,
		TopK:          s.config.RAGConfig.TopK,
		IsInitialized: s.isInitialized,
	}

	s.respondJSON(w, response)
}

func (s *Server) handleConfigUpdate(w http.ResponseWriter, r *http.Request) {
	if !s.isInitialized {
		http.Error(w, "System not initialized", http.StatusPreconditionFailed)
		return
	}

	var req ConfigUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Update configuration
	if req.TopK > 0 && req.TopK <= 20 {
		s.config.RAGConfig.TopK = req.TopK
	}
	if req.Model != "" {
		s.config.RAGConfig.GenerationModel = req.Model
	}
	if req.Temperature >= 0 && req.Temperature <= 1 {
		s.config.RAGConfig.Temperature = float32(req.Temperature)
	}

	// Update RAG system config
	s.ragSystem.UpdateConfig(s.config.RAGConfig)

	s.respondJSON(w, map[string]interface{}{
		"success": true,
		"message": "Configuration updated successfully",
		"config": map[string]interface{}{
			"topk":        s.config.RAGConfig.TopK,
			"model":       s.config.RAGConfig.GenerationModel,
			"temperature": s.config.RAGConfig.Temperature,
		},
	})
}

func (s *Server) handleSimilar(w http.ResponseWriter, r *http.Request) {
	if !s.isInitialized {
		http.Error(w, "System not initialized", http.StatusPreconditionFailed)
		return
	}

	var req SimilarRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "Query cannot be empty", http.StatusBadRequest)
		return
	}

	topK := req.TopK
	if topK <= 0 {
		topK = s.config.RAGConfig.TopK
	}

	ctx := context.Background()
	results, err := s.ragSystem.GetSimilarChunks(ctx, req.Query, topK)

	response := SimilarResponse{}

	if err != nil {
		response.Success = false
		response.Error = err.Error()
	} else {
		response.Success = true
		for _, result := range results {
			response.Results = append(response.Results, SourceInfo{
				Title:   result.Chunk.Title,
				URL:     result.Chunk.SourceURL,
				Score:   float64(result.Score),
				Content: result.Chunk.Content[:min(300, len(result.Chunk.Content))],
			})
		}
	}

	s.respondJSON(w, response)
}

func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Handle WebSocket communication for real-time updates
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}
		// Echo back or handle specific messages
	}
}

func (s *Server) runScrapeProcess(url string) {
	log.Printf("Starting scrape process for: %s", url)

	// Initialize scraper with current config
	scraper, err := scraper.NewWebScraper(s.config.ScraperConfig)
	if err != nil {
		log.Printf("Failed to create scraper: %v", err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Start scraping
	if err := scraper.Start(ctx); err != nil {
		log.Printf("Scraping failed: %v", err)
		return
	}

	content := scraper.GetContent()
	log.Printf("Scraping completed. Found %d unique content items", len(content))

	// Process and chunk the scraped content
	if err := chunking.ProcessAndSave(s.config.ChunkingConfig, content, s.config.ScraperConfig.OutDir); err != nil {
		log.Printf("Failed to process and save chunks: %v", err)
		return
	}

	// Embed the scraped content
	ollamaUrl := "http://localhost:11434"
	chunksFile := s.config.ScraperConfig.OutDir + "/chunks_all.json"

	if err := embedding.ProcessChunksToEmbeddings(chunksFile, s.config.OutputDir, ollamaUrl, s.config.EmbeddingConfig); err != nil {
		log.Printf("Failed to process chunks to embeddings: %v", err)
		return
	}

	// Update vector database
	embeddedChunksFile := s.config.OutputDir + "/embedded_chunks_all.json"
	vdb, err := vectordb.SetupVectorDatabase(embeddedChunksFile, vectordb.DefaultVectorDBConfig())
	if err != nil {
		log.Printf("Failed to setup vector database: %v", err)
		return
	}
	defer vdb.Close()

	log.Println("Scraping and indexing completed successfully!")
}

func (s *Server) updateStats(processingTime int64, tokens int, success bool) {
	s.stats.mu.Lock()
	defer s.stats.mu.Unlock()

	s.stats.QueryCount++
	s.stats.TotalTime += processingTime
	s.stats.TokensGenerated += int64(tokens)
	if success {
		s.stats.SuccessCount++
	}
}

func (s *Server) respondJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func (s *Server) Close() {
	if s.ragSystem != nil {
		s.ragSystem.Close()
	}
}

// Utility functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// StartServer starts the HTTP server
func StartServer(config config.Config, port string) error {
	if port == "" {
		port = "8080"
	}

	// Setup logging
	logHandler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})
	slog.SetDefault(slog.New(logHandler))

	// Create and start server
	server := NewServer(config)
	defer server.Close()

	handler := server.setupRoutes()

	log.Printf("🚀 RAG API Server starting on port %s", port)
	log.Printf("📱 Web interface available at: http://localhost:%s", port)
	log.Printf("🔌 API endpoints available at: http://localhost:%s/api/", port)

	return http.ListenAndServe(":"+port, handler)
}