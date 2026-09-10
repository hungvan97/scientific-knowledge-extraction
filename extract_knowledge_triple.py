import json
import os
from pathlib import Path
from typing import List

import torch
from bs4 import BeautifulSoup
from pydantic import BaseModel, TypeAdapter, ValidationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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


def extract_text_from_tei_xml(tei_path: str) -> str:
    """Extract readable text from a TEI-XML article file."""
    try:
        with open(tei_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file.read(), "lxml-xml")

        text_node = soup.find("text") or soup.find("body")
        if text_node is None:
            return ""

        text = text_node.get_text(separator="\n")
        return " ".join(text.split())
    except Exception as e:
        print(f"Error parsing {tei_path}: {e}")
        return ""

def build_prompt(text: str) -> str:
    """Construct a prompt that asks the LLM to return structured triples."""
    return f"""You are an expert in psychology and computational knowledge representation.

Your task is to extract key scientific information from psychology research articles to build a structured knowledge graph.

The knowledge graph aims to represent the relationships between psychological topics or constructs and their associated measurement instruments or scales. Specifically, for each article, extract information in the form of triples that capture:
1) The psychological topic or construct being studied
2) The measurement instrument or scale used to assess it
3) A brief justification of no more than 1 sentence from the article text supporting this measurement link

Guidelines:
- Extract meaningful phrases (not full sentences or vague descriptions) for both `topic_or_construct` and `measured_by`, suitable for inclusion in a knowledge graph.
- Include a short justification for each extraction that clearly supports the connection.
- If the article does not discuss psychological constructs and how they are measured (for example, no mention of constructs, instruments, or scales), return an empty list `[]`.

Input Paper:
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
    text = extract_text_from_tei_xml(file_path)
    if not text:
        print(f"⚠️ Skipping empty or malformed text in {file_path}")
        return False

    prompt = build_prompt(text[:40000])
    prompt_path = Path(prompt_dir) / f"{Path(file_path).stem}.prompt.txt"
    output_path = Path(output_dir) / f"{Path(file_path).stem}.json"

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    try:
        try:
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
            print(f"✅ {file_path} | Extracted 0 triples")
            return False

        serializable = []
        for item in results:
            if hasattr(item, "model_dump"):
                serializable.append(item.model_dump())
            elif hasattr(item, "dict"):
                serializable.append(item.dict())
            else:
                serializable.append(dict(item))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        print(f"✅ {file_path} | Extracted {len(results)} triples")
        return True

    except ValidationError as ve:
        print(f"❌ Validation failed for {file_path}:\n{ve}")
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage = True
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        return_full_text=False,
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

        generated = text_gen_pipeline(
            [prompt],
            max_new_tokens=512,
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
