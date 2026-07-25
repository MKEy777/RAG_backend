import pypdfium2 as pdfium
try:
    print("version:", pdfium.__version__)
except AttributeError:
    import importlib.metadata
    print("version:", importlib.metadata.version("pypdfium2"))

print("has FPDF_PAGEOBJ_IMAGE:", hasattr(pdfium, "FPDF_PAGEOBJ_IMAGE"))

# Check what constants exist
consts = [x for x in dir(pdfium) if "OBJ" in x.upper() or "PAGEOBJ" in x.upper()]
print("OBJ constants:", consts)

# Check all uppercase constants
all_consts = [x for x in dir(pdfium) if x.isupper() and not x.startswith('_')]
print("All uppercase:", all_consts[:30])

# Try the raw int constant for image objects (FPDF_PAGEOBJ_IMAGE = 3 in pdfium)
print("\nTrying raw filter with int 3 (IMAGE)...")
doc = pdfium.PdfDocument.new()
page = doc.new_page(100, 100)
try:
    objs = list(page.get_objects(filter=(3,)))
    print("  filter=(3,) works, objects:", len(objs))
except Exception as e:
    print("  filter=(3,) failed:", e)

try:
    objs = list(page.get_objects())
    print("  no filter works, objects:", len(objs))
except Exception as e:
    print("  no filter failed:", e)
