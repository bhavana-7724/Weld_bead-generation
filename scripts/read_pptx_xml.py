import zipfile
import xml.etree.ElementTree as ET

pptx_path = r"c:\Users\cool\OneDrive\Desktop\Project\weld_project\FINAL_PROJECT_PRESENTATION.pptx"
with zipfile.ZipFile(pptx_path, 'r') as z:
    slide_files = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
    slide_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(f"Total slides found: {len(slide_files)}")
    for i, s in enumerate(slide_files, 1):
        xml_content = z.read(s)
        tree = ET.fromstring(xml_content)
        texts = [node.text for node in tree.iter() if node.text and node.text.strip()]
        title = texts[0] if texts else "Empty"
        snippet = " | ".join(texts[:3])
        print(f"Slide {i}: {snippet[:90]}")
