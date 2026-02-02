.PHONY: build clean test

build:
	go build -o sgf_parser main.go
	@echo "Built sgf_parser binary"

build-release:
	CGO_ENABLED=0 go build -ldflags="-s -w" -o sgf_parser main.go
	@echo "Built optimized sgf_parser binary"

clean:
	rm -f sgf_parser

test:
	go test -v ./...

install:
	go build -o sgf_parser main.go
	@echo "Binary built at ./sgf_parser"
	@echo "Add this to your PATH or use directly"
