# Multi-Language Poker Analysis System

This system combines the strengths of Go and Python:

- **Go** - Scalable concurrent hand history processing service
- **Python** - Machine learning pipeline and system orchestration

## Project Structure

```
multi-poker/
├── go-services/        # Go concurrent services
└── python-orchestrator # Python ML and integration layer
```

## Building the Project

1. Install dependencies:
   - Rust (latest stable)
   - Go 1.20+
   - Python 3.8+

2. Build instructions:
   ```
    # Build Go services
    cd go-services && go build

   # Install Python dependencies
    cd python-orchestrator && pip install -r requirements.txt
   ```
   
## Components

1. Go Services
    - Hand history processing service
    - Concurrent game state management
    - Real-time data processing
    - Hand evaluation

2. Python Orchestrator
    - Machine learning pipeline
    - Service coordination
    - Data preprocessing and analysis
