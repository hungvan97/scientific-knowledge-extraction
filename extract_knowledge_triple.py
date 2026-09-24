import json
import os
from pathlib import Path
from typing import List

import torch

import json
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from pydantic import BaseModel, TypeAdapter, ValidationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

try:
    import instructor
    from instructor import Instructor
except ImportError as exc:
    raise ImportError(
        "The 'instructor' package is required. Install it with: pip install instructor"
    ) from exc


class PsychTriple(BaseModel):
    topic_or_construct: str
    measured_by: str
    justification: str


def extract_text_from_tei_xml(tei_path: str) -> list[dict]:
    """Extract readable text from a TEI-XML article file."""
    tree = ET.parse(tei_path)
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
    return chunks

def build_prompt(text: str) -> str:
    """Construct a prompt that asks the LLM to return structured triples."""
    return f"""You are an expert in psychology and computational knowledge representation.

Your task is to extract key scientific information from the following input text, which extracted from psychology research articles, to build a structured knowledge graph.

The knowledge graph aims to represent the relationships between psychological topics or constructs and their associated measurement instruments or scales. Specifically, for each article, extract information in the form of triples that capture:
1) The psychological topic or construct being studied
2) The measurement instrument or scale used to assess it
3) A brief justification of no more than 1 sentence from the article text supporting this measurement link

Guidelines:
- Extract meaningful phrases (not full sentences or vague descriptions) for both `topic_or_construct` and `measured_by`, suitable for inclusion in a knowledge graph.
- Include a short justification for each extraction that clearly supports the connection.
- If the article does not discuss psychological constructs and how they are measured (for example, no mention of constructs, instruments, or scales), return an empty list `[]`.

Input text to be processed:
\"\"\"{text}\"\"\"

Output: Provide your response as a JSON list in the following format:
[
  {{
    "topic_or_construct": "...",
    "measured_by": "...",
    "justification": "..."
  }}
]

Return only the JSON array. Do not include explanations, analysis, or Markdown.
""".strip()


def process_file(file_path: str, prompt_dir: str, output_dir: str, llm) -> bool:
    """Process one TEI XML file and save the extracted triples to JSON."""
    chunk_text = extract_text_from_tei_xml(file_path)
    if not chunk_text:
        print(f"⚠️ Skipping empty or malformed text in {file_path}")
        return False

    for index, chunk in enumerate(chunk_text):
        text = chunk['heading'] + ':\n\n' + "\n\n".join(chunk['text'])
        prompt = build_prompt(text)
        prompt_path = Path(prompt_dir) / f"{Path(file_path).stem}.{index+1}.prompt.txt"

        output_path = Path(output_dir) / f"{Path(file_path).stem}.chunk{index+1}.json"
        all_triples: list[dict] = []

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        try:
            try:
                print(f"Text to process: {text}")
                results = llm.create(
                    response_model=List[PsychTriple],
                    messages=[{"role": "user", "content": prompt}],
                )
            except TypeError:
                try:
                    results = llm.create(
                        response_model=List[PsychTriple],
                        messages=[{"role": "user", "content": prompt}],
                        mode=instructor.Mode.JSON,
                    )
                except TypeError:
                    results = llm.create(
                        response_model=List[PsychTriple],
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )

            if not results:
                print(f"❌ Chunk number {index+1} extracts 0 triple")
                continue
            else:
                print(f"✅ Chunk number {index+1} extracts {len(results)} triples")
                for triple in results:
                    if hasattr(triple, "model_dump"):
                        all_triples.append(triple.model_dump())
                    elif hasattr(triple, "dict"):
                        all_triples.append(triple.dict())
                    else:
                        all_triples.append(dict(triple))

        except ValidationError as ve:
            print(f"❌ Chunk {index+1} failed to validate:\n{ve}")
        except Exception as e:
            print(f"❌ Chunk {index+1} failed to process: {e}")
        

        if all_triples:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_triples, f, ensure_ascii=False, indent=2)

    return False


def main():
    cur_dir = os.getcwd()
    input_dir = "/home/dokka/Desktop/qwen-env/Bachelor Thesis"
    prompt_dir = input_dir + "/inp_prompt"          # input directory with .xml files
    output_dir = input_dir + "/out_json"            # output directory to save JSON
    model_name = "Qwen/Qwen2.5-3B-Instruct"         # Hugging Face model name
    print(prompt_dir)

    # input_dir = input("Enter input directory with .xml files: ").strip()
    # prompt_dir = input("Enter directory to save prompts: ").strip()
    # output_dir = input("Enter directory to save JSON outputs: ").strip()
    # model_name = input("Enter Hugging Face model name (e.g., Qwen/Qwen2.5-3B-Instruct): ").strip()

    for d in [input_dir, prompt_dir, output_dir]:
        os.makedirs(d, exist_ok=True)

    # Model configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage = True
    )
    generation_config = GenerationConfig(
        max_new_tokens=2048,
        max_length=None,
    )
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
    )

    def hf_create(*, response_model=None, messages=None, **kwargs):
        prompt = ""
        if isinstance(messages, list) and messages:
            first_message = messages[0]
            if isinstance(first_message, dict):
                prompt = str(first_message.get("content", ""))
        elif isinstance(messages, str):
            prompt = messages

        if not prompt:
            raise ValueError("Empty prompt was provided to the Hugging Face pipeline.")

        ## Formatted prompts using Qwen’s chat template before text generation. Added the user role and generation prompt marker
        chat_messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        generated = text_gen_pipeline(
            [formatted_prompt],
            generation_config = generation_config,
            return_full_text=False,
        )

        generated_value: object = generated
        if (
            isinstance(generated_value, list)
            and generated_value
            and isinstance(generated_value[0], list)
        ):
            generated_value = generated_value[0]

        if isinstance(generated_value, list) and generated_value:
            first_item = generated_value[0]
            if isinstance(first_item, dict):
                text = str(first_item.get("generated_text", ""))
            else:
                text = str(first_item)
        else:
            text = str(generated_value)

        try:
            decoder = json.JSONDecoder()

            for index, character in enumerate(text):
                if character != "[":
                    continue

                try:
                    parsed, _ = decoder.raw_decode(text[index:])
                    adapter = TypeAdapter(response_model)
                    return adapter.validate_python(parsed)
                except (json.JSONDecodeError, ValidationError):
                    continue

            raise ValueError("No valid JSON array found")

        except Exception as exc:
            raise ValueError(
                f"Model did not return valid JSON: {text[:500]}"
            ) from exc

    llm = Instructor(client=None, create=hf_create, mode=instructor.Mode.JSON)

    files = sorted(Path(input_dir).glob("*.xml"))
    if not files:
        print(f"⚠️ No .xml files found in {input_dir}")
        return

    total_articles = 0
    satisfied_articles = 0

    for xml_file in tqdm(files, desc="Processing TEI XML files"):
        total_articles += 1
        success = process_file(str(xml_file), prompt_dir, output_dir, llm)
        if success:
            satisfied_articles += 1

    print("\n=== Extraction Summary ===")
    print(f"Total articles processed: {total_articles}")
    print(f"Articles with extracted constructs and measures: {satisfied_articles}")
    print(f"Articles without relevant content: {total_articles - satisfied_articles}")


if __name__ == "__main__":
    main()
