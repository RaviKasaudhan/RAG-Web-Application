# RAG Web Application

## Introduction

The RAG Web Application leverages Azure and OpenAI services to process and index documents, providing summarized answers using the "text-embedding-ada-003" model.

## Getting Started

### Prerequisites

To use this application, you will need the following:

1. **Azure Blob Storage**:
   - **Two Blob Containers**: 
     - One for storing your files.
     - One for storing the index of your files.

2. **Credentials**:
   - **OpenAI**: For utilizing the "text-embedding-ada-003" model.
   - **Azure Form Recognizer**: For extracting data from documents.

### Setup Instructions

1. **Clone the Repository**:
   First, clone this repository to your local directory.
   ```bash
   git clone https://github.com/YourUsername/RAG-Web-Application.git
