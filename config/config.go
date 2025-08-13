package config

import (
	"lcbchat/chunking"
	"lcbchat/embedding"
	"lcbchat/rag"
	"lcbchat/scraper"
)

// Config holds all configuration for the application
type Config struct {
	RAGConfig       rag.RAGConfig
	OutputDir       string
	ScraperConfig   scraper.ScraperConfig
	ChunkingConfig  chunking.ChunkingConfig
	EmbeddingConfig embedding.EmbeddingConfig
}

// DefaultConfig returns a default configuration
func DefaultConfig() Config {
	return Config{
		RAGConfig:       rag.DefaultRAGConfig(),
		OutputDir:       "./output",
		ScraperConfig:   scraper.DefaultScraperConfig(),
		ChunkingConfig:  chunking.DefaultChunkingConfig(),
		EmbeddingConfig: embedding.DefaultEmbeddingConfig(),
	}
}