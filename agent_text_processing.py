class VectorMemoryRAGPlugin:
    def __init__(self):
        self.text_chunks = []
        self.index = None
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, doc_text: str, chunk_size: int = 500):
        self.text_chunks = [
            doc_text[i:i + chunk_size]
            for i in range(0, len(doc_text), chunk_size)
        ]
        vectors = self.embeddings.encode(self.text_chunks, convert_to_numpy=True)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    @kernel_function(description="retrieve relevant chunks from uploaded claim documents.")
    async def retrieve_chunks(self, query: Annotated[str, "Query to summmarise / retrieve relevant claim information"]) -> str:
        if not self.index:
            return "No documents indexed yet."
        query_vec = self.embeddings.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, k=3)
        relevant_chunks = [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
        return "\n---\n".join(relevant_chunks)


class StructureClaimData:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @kernel_function(description="Return a JSON containing structured claim_data, use before calling other plugins")
    async def StructureClaimData(self, claim_text: Annotated[str, "The unstructured claim_text string input"]) -> str:
        prompt = f"""
Extract the following fields from the text below. If a field is not present, leave it blank. for coverage_amount, extract the number. For example USD 150,000,000 should be coverage_amount: 150000000

Required fields (as JSON):
{{
    "organisation_name": "",
    "region_of_operation": "",
    "coverage_amount": "",
    "premium": "",
    "export_destination": "",
    "client_priorities": ""
}}

Text:
\"\"\"{claim_text}\"\"\"

Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.
"""
        completion = await self.kernel.invoke_prompt(prompt)
        if hasattr(completion, "result"):
            return str(completion.result).strip()
        return str(completion).strip()
