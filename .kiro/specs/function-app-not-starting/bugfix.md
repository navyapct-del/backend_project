# Bugfix Requirements Document

## Introduction

The Azure Function App in `azure_upload_function/` fails to start when launched with the standard `func start` command. The host process picks up the system-default Python (3.14) instead of the project-configured Python 3.10 environment where all dependencies are installed. This causes an immediate import failure at startup because none of the required packages (`azure-functions`, `openai`, `pandas`, etc.) are available in the system Python environment.

A secondary bug exists in `delete_service.py`: the search index name is hardcoded as `"documents-index"`, while the actual index created and used by `search_service.py` is `"documents-index-v2"`. This means every document deletion silently fails to remove the document from Azure AI Search, leaving orphaned search entries.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the developer runs `func start` from the `azure_upload_function/` directory THEN the Azure Functions host selects the system Python (3.14) instead of the project-configured Python 3.10, causing all service imports to fail with `ModuleNotFoundError` and the function app to crash at startup

1.2 WHEN the function app starts with the wrong Python runtime THEN the system raises `STARTUP IMPORT ERROR` and re-raises the exception, preventing any HTTP routes from being registered

1.3 WHEN a document is deleted via the `DELETE /documents/{id}` endpoint THEN the system sends the delete request to the index named `"documents-index"` instead of `"documents-index-v2"`, so the document remains in the active search index and continues to appear in search results after deletion

### Expected Behavior (Correct)

2.1 WHEN the developer runs `func start` from the `azure_upload_function/` directory THEN the Azure Functions host SHALL use the Python 3.10 runtime (matching `.python-version`) so that all installed packages are available and the function app starts successfully

2.2 WHEN the function app starts with the correct Python runtime THEN the system SHALL successfully import all services and register all HTTP routes without errors

2.3 WHEN a document is deleted via the `DELETE /documents/{id}` endpoint THEN the system SHALL send the delete request to the index named `"documents-index-v2"` so the document is removed from search results

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the developer runs `func start` using the provided `start.ps1` script THEN the system SHALL CONTINUE TO start successfully using Python 3.10 as it does today

3.2 WHEN a valid file is uploaded via `POST /upload` THEN the system SHALL CONTINUE TO process, index, and store the document without errors

3.3 WHEN a search query is submitted via `POST /query` THEN the system SHALL CONTINUE TO return results from the `"documents-index-v2"` index

3.4 WHEN the `GET /health` endpoint is called THEN the system SHALL CONTINUE TO return the environment variable status without errors

3.5 WHEN a document is deleted and the raw blob, text blob, and structured-data blob exist THEN the system SHALL CONTINUE TO delete all three blobs from Azure Blob Storage as part of the cascade delete
