import argparse
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig, TrainingArguments, Trainer,
                           DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "### De-identified Clinical Input:\n{input}\n\n"
    "### Structured SOAP Note:\n{output}"
)

def build_dataset(train_file, tokenizer, max_len=1024):
    ds = load_dataset("json", data_files=train_file, split="train")

    def _tok(ex):
        text = PROMPT_TEMPLATE.format(input=ex["input"], output=ex["output"])
        out = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    return ds.map(_tok, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", default="checkpoints/clinical-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)          # LoRA rank
    ap.add_argument("--alpha", type=int, default=32)       # LoRA scaling
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # W' = W_0 + BA, with B in R^(d x r), A in R^(r x k), r << min(d, k)
    lora_config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(args.train_file, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                       data_collator=collator)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
