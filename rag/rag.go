package rag

// The RAG package should handle:

// Taking user queries
// Embedding the query
// Searching the vector database
// Generating responses with the LLM
// Combining everything into a coherent answer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"lcbchat/embedding"
	"lcbchat/vectordb"
)

// RAGConfig holds configuration for the RAG system
type RAGConfig struct {
	// Ollama settings
	OllamaURL       string
	EmbeddingModel  string
	GenerationModel string
	
	// Vector DB settings
	VectorDBConfig vectordb.VectorDBConfig
	
	// Retrieval settings
	TopK           int     // Number of chunks to retrieve
	ScoreThreshold float32 // Minimum similarity score
	
	// Generation settings
	MaxTokens      int
	Temperature    float32
	SystemPrompt   string
}

// DefaultRAGConfig returns sensible defaults
func DefaultRAGConfig() RAGConfig {
	return RAGConfig{
		OllamaURL:       "http://localhost:11434",
		EmbeddingModel:  "nomic-embed-text",
		GenerationModel: "llama3.1:8b",
		VectorDBConfig:  vectordb.DefaultVectorDBConfig(),
		TopK:           5,
		ScoreThreshold: 0.3,
		MaxTokens:      2000,
		Temperature:    0.7,
		SystemPrompt: `You are a helpful AI assistant that answers questions based on the provided context. 
Use only the information from the context to answer questions. If the context doesn't contain 
enough information to answer the question, say so clearly. Be as thorough as possible.`,
	}
}

// RAGSystem orchestrates the entire RAG pipeline
type RAGSystem struct {
	config            RAGConfig
	embeddingClient   *embedding.OllamaEmbeddingClient
	vectorDB          *vectordb.VectorDatabase
	generationClient  *OllamaGenerationClient
}

// QueryResult represents the result of a RAG query
type QueryResult struct {
	Query           string                    `json:"query"`
	Answer          string                    `json:"answer"`
	Sources         []vectordb.SearchResult  `json:"sources"`
	ProcessingTime  time.Duration            `json:"processing_time"`
	RetrievalTime   time.Duration            `json:"retrieval_time"`
	GenerationTime  time.Duration            `json:"generation_time"`
	TokensGenerated int                      `json:"tokens_generated,omitempty"`
}

// OllamaGenerationClient handles text generation with Ollama
type OllamaGenerationClient struct {
	BaseURL    string
	HTTPClient *http.Client
	Model      string
}

// OllamaGenerationRequest represents a generation request to Ollama
type OllamaGenerationRequest struct {
	Model       string                 `json:"model"`
	Prompt      string                 `json:"prompt"`
	Stream      bool                   `json:"stream"`
	Options     map[string]any `json:"options,omitempty"`
	System      string                 `json:"system,omitempty"`
}

// OllamaGenerationResponse represents a generation response from Ollama
type OllamaGenerationResponse struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Response           string    `json:"response"`
	Done               bool      `json:"done"`
	Context            []int     `json:"context,omitempty"`
	TotalDuration      int64     `json:"total_duration,omitempty"`
	LoadDuration       int64     `json:"load_duration,omitempty"`
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int64     `json:"eval_duration,omitempty"`
}

// NewRAGSystem creates a new RAG system
func NewRAGSystem(config RAGConfig) (*RAGSystem, error) {
	// Initialize embedding client
	embeddingConfig := embedding.DefaultEmbeddingConfig()
	embeddingConfig.Model = config.EmbeddingModel
	embeddingClient := embedding.NewOllamaEmbeddingClient(config.OllamaURL, config.EmbeddingModel)

	// Initialize vector database
	vectorDB, err := vectordb.NewVectorDatabase(config.VectorDBConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize vector database: %w", err)
	}

	// Initialize generation client
	generationClient := &OllamaGenerationClient{
		BaseURL: config.OllamaURL,
		Model:   config.GenerationModel,
		HTTPClient: &http.Client{
			Timeout: 120 * time.Second, // Generous timeout for generation
		},
	}

	return &RAGSystem{
		config:           config,
		embeddingClient:  embeddingClient,
		vectorDB:         vectorDB,
		generationClient: generationClient,
	}, nil
}

// Query performs a complete RAG query
func (r *RAGSystem) Query(ctx context.Context, query string) (*QueryResult, error) {
	startTime := time.Now()

	// Step 1: Generate embedding for the query
	log.Printf("Generating embedding for query: %s", query)
	embedStart := time.Now()
	
	queryEmbedding, err := r.embeddingClient.GenerateEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}
	embedTime := time.Since(embedStart)
	
	// Step 2: Search vector database
	log.Printf("Searching vector database for similar content...")
	searchStart := time.Now()
	
	searchResults, err := r.vectorDB.Search(
		ctx, 
		queryEmbedding, 
		uint64(r.config.TopK), 
		nil, // No additional filters for now
	)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector database: %w", err)
	}
	
	// Filter results by score threshold
	var filteredResults []vectordb.SearchResult
	for _, result := range searchResults {
		if result.Score >= r.config.ScoreThreshold {
			filteredResults = append(filteredResults, result)
		}
	}
	
	retrievalTime := time.Since(searchStart)
	log.Printf("Retrieved %d relevant chunks (filtered from %d)", len(filteredResults), len(searchResults))

	if len(filteredResults) == 0 {
		return &QueryResult{
			Query:          query,
			Answer:         "I couldn't find any relevant information to answer your question.",
			Sources:        []vectordb.SearchResult{},
			ProcessingTime: time.Since(startTime),
			RetrievalTime:  retrievalTime,
		}, nil
	}

	// Step 3: Build context from retrieved chunks
	context_text := r.buildContext(filteredResults)
	
	// Step 4: Generate response
	log.Printf("Generating response using %s...", r.config.GenerationModel)
	genStart := time.Now()
	
	response, tokensGenerated, err := r.generateResponse(ctx, query, context_text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}
	
	generationTime := time.Since(genStart)
	totalTime := time.Since(startTime)

	log.Printf("RAG query completed in %v (embedding: %v, retrieval: %v, generation: %v)", 
		totalTime, embedTime, retrievalTime, generationTime)

	return &QueryResult{
		Query:           query,
		Answer:          response,
		Sources:         filteredResults,
		ProcessingTime:  totalTime,
		RetrievalTime:   retrievalTime,
		GenerationTime:  generationTime,
		TokensGenerated: tokensGenerated,
	}, nil
}

// buildContext creates a formatted context string from search results
func (r *RAGSystem) buildContext(results []vectordb.SearchResult) string {
	var contextBuilder strings.Builder
	
	contextBuilder.WriteString("Context:\n")
	
	for i, result := range results {
		contextBuilder.WriteString(fmt.Sprintf("\n[Source %d - Score: %.3f]\n", i+1, result.Score))
		contextBuilder.WriteString(fmt.Sprintf("Title: %s\n", result.Chunk.Title))
		contextBuilder.WriteString(fmt.Sprintf("URL: %s\n", result.Chunk.SourceURL))
		contextBuilder.WriteString(fmt.Sprintf("Content: %s\n", result.Chunk.Content))
		
		if i < len(results)-1 {
			contextBuilder.WriteString("\n---\n")
		}
	}
	
	return contextBuilder.String()
}

// generateResponse generates a response using the Ollama generation model
func (r *RAGSystem) generateResponse(ctx context.Context, query, context string) (string, int, error) {
	prompt := fmt.Sprintf(`%s

%s

Question: %s

Please provide a helpful and accurate answer based on the context above.`, 
		r.config.SystemPrompt, context, query)

	request := OllamaGenerationRequest{
		Model:  r.config.GenerationModel,
		Prompt: prompt,
		Stream: false, // We want the complete response
		System: r.config.SystemPrompt,
		Options: map[string]interface{}{
			"num_predict":  r.config.MaxTokens,
			"temperature": r.config.Temperature,
			"top_p":       0.9,
			"stop":        []string{"Question:", "Context:", "\n\nQuestion:"},
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", r.config.OllamaURL+"/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", 0, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := r.generationClient.HTTPClient.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", 0, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var response OllamaGenerationResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", 0, fmt.Errorf("failed to decode response: %w", err)
	}

	return strings.TrimSpace(response.Response), response.EvalCount, nil
}

// QueryWithFilters performs a RAG query with additional vector database filters
func (r *RAGSystem) QueryWithFilters(ctx context.Context, query string, filters map[string]any) (*QueryResult, error) {
	startTime := time.Now()

	// Generate embedding for the query
	queryEmbedding, err := r.embeddingClient.GenerateEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Search with filters
	searchStart := time.Now()
	searchResults, err := r.vectorDB.Search(ctx, queryEmbedding, uint64(r.config.TopK), filters)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector database: %w", err)
	}

	// Filter by score and continue with normal flow
	var filteredResults []vectordb.SearchResult
	for _, result := range searchResults {
		if result.Score >= r.config.ScoreThreshold {
			filteredResults = append(filteredResults, result)
		}
	}

	retrievalTime := time.Since(searchStart)

	if len(filteredResults) == 0 {
		return &QueryResult{
			Query:          query,
			Answer:         "I couldn't find any relevant information matching your criteria.",
			Sources:        []vectordb.SearchResult{},
			ProcessingTime: time.Since(startTime),
			RetrievalTime:  retrievalTime,
		}, nil
	}

	// Build context and generate response
	context_text := r.buildContext(filteredResults)
	
	genStart := time.Now()
	response, tokensGenerated, err := r.generateResponse(ctx, query, context_text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}

	return &QueryResult{
		Query:           query,
		Answer:          response,
		Sources:         filteredResults,
		ProcessingTime:  time.Since(startTime),
		RetrievalTime:   retrievalTime,
		GenerationTime:  time.Since(genStart),
		TokensGenerated: tokensGenerated,
	}, nil
}

// GetSimilarChunks returns similar chunks without generating a response
func (r *RAGSystem) GetSimilarChunks(ctx context.Context, query string, limit int) ([]vectordb.SearchResult, error) {
	queryEmbedding, err := r.embeddingClient.GenerateEmbedding(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	results, err := r.vectorDB.Search(ctx, queryEmbedding, uint64(limit), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector database: %w", err)
	}

	return results, nil
}

// UpdateConfig allows updating configuration at runtime
func (r *RAGSystem) UpdateConfig(config RAGConfig) {
	r.config = config
	// Note: This doesn't reinitialize clients - you'd need to recreate the RAGSystem for that
}

// Close closes all connections
func (r *RAGSystem) Close() error {
	if r.vectorDB != nil {
		return r.vectorDB.Close()
	}
	return nil
}