"""
This script extracts chunks of text from a TEI XML file and saves them as a JSON file.
It performs the following steps:
1. Parse the TEI XML file.
2. Iterate through the <div> elements in the XML.
3. For each <div>, extract the heading from the <head> element and the text from the <p> elements.
4. Create a list of dictionaries, each containing a chunk ID, heading, and text.
5. Save the list of chunks to a JSON file.

Usage:
python extract_chunks.py <path_to_tei_xml_file>
"""
import json
import xml.etree.ElementTree as ET

xml_path = (
    "./Bachelor Thesis/"
    "Hepper2015self-esteemencyclopediaofmentalhealthchapter.pdf.tei.xml"
)

tree = ET.parse(xml_path)
root = tree.getroot()

chunks = []

for div in root.iter():
    if div.tag.rsplit("}", 1)[-1] != "div":
        continue

    heading = None
    paragraphs = []

    for child in div:
        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "head":
            heading = " ".join(child.itertext()).strip()

        elif tag == "p":
            text = " ".join(child.itertext()).strip()
            if text:
                paragraphs.append(text)

    if heading and paragraphs:
        chunks.append({
            "chunk_id": len(chunks),
            "heading": heading,
            "text": paragraphs
        })

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print(f"Saved {len(chunks)} chunks to {file.name}")