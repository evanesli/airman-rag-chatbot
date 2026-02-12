from src.ingestion.pdf_loader import AviationPDFLoader

# Initialize loader
loader = AviationPDFLoader(
    extraction_method="auto",
    remove_headers_footers=True,
    clean_text=True
)

# Load your aviation PDF
pdf_path = "data/raw/your_aviation_manual.pdf"  # Update this!
pages, metadata = loader.load_pdf(pdf_path)

# Print results
print("="*60)
print(f"📄 Document: {metadata.filename}")
print(f"📊 Pages: {metadata.total_pages}")
print(f"💾 Size: {metadata.file_size_mb} MB")
print(f"✍️  Author: {metadata.author}")
print("="*60)

# Show first 3 pages
for i in range(min(3, len(pages))):
    page = pages[i]
    print(f"\n--- Page {page.page_number} ---")
    print(f"Characters: {len(page.text)}")
    print(f"Preview: {page.text[:200]}...")
    print()

# Calculate statistics
total_chars = sum(len(p.text) for p in pages)
avg_chars = total_chars / len(pages)

print(f"📈 Statistics:")
print(f"   Total characters: {total_chars:,}")
print(f"   Average per page: {avg_chars:.0f}")
print(f"   Shortest page: {min(len(p.text) for p in pages)} chars")
print(f"   Longest page: {max(len(p.text) for p in pages)} chars")