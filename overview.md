**Chunking** is the process of splitting large documents into smaller pieces to improve retrieval and processing by the model.
**Chunking is needed** to Handles large PDFs, to improves retrieval accuracy, to fits within LLM context limits, to reduces irrelevant information
We use RecursiveCharacterTextSplitter to divide documents into chunks.
**Parameters we are using**
chunk_size = 500(size of each chunk)
chunk_overlap = 100(shared content between chunks so that no data left)

**challenges faced**
During the implementation of the multiple file upload feature, I faced an issue in FastAPI where the /upload endpoint was not behaving as expected in Swagger UI. Instead of showing a proper file upload option, it was displaying the input field as a string type, which prevented selecting and uploading multiple PDF files correctly. This created confusion because the backend logic was already designed to handle multiple files using List[UploadFile]. After analyzing the issue, I understood that this was a limitation or improper rendering of multipart file inputs in Swagger UI rather than a backend problem. To overcome this, I used Postman for testing the API. In Postman, I configured the request using form-data, added multiple files fields, and set each field type to “File,” which allowed me to successfully upload multiple PDFs in a single request. This confirmed that the backend implementation was correct, and the issue was only with the UI representation in FastAPI’s Swagger interface.