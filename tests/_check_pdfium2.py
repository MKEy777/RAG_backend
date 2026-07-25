import pypdfium2 as pdfium
from pathlib import Path

pdf_path = list(Path(__file__).parent.glob("*.pdf"))[0]
doc = pdfium.PdfDocument(str(pdf_path))
print(f"Total pages: {len(doc)}")

# Find a page with image objects
for i in range(min(len(doc), 20)):
    page = doc[i]
    objs = list(page.get_objects(filter=(3,)))
    if objs:
        print(f"\nPage {i}: {len(objs)} image objects")
        obj = objs[0]
        print(f"Type: {type(obj)}")
        methods = [m for m in dir(obj) if not m.startswith('_')]
        print(f"Methods: {methods}")
        if hasattr(obj, 'get_matrix'):
            m = obj.get_matrix()
            print(f"get_matrix(): {m}")
            print(f"  matrix attrs: {[a for a in dir(m) if not a.startswith('_')]}")
        if hasattr(obj, 'matrix'):
            print(f"matrix: {obj.matrix}")
        break
else:
    print("No image objects found in first 20 pages")
    # Check generic object methods on page 0
    page = doc[0]
    all_objs = list(page.get_objects())
    if all_objs:
        obj = all_objs[0]
        print(f"\nFirst object type: {type(obj)}")
        methods = [m for m in dir(obj) if not m.startswith('_')]
        print(f"Methods: {methods}")

doc.close()
