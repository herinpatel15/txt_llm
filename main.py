from transformers import TextDataset, GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )


def main():
    modle_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(modle_name)
    modle = GPT2LMHeadModel.from_pretrained(modle_name)

    train_dataset = load_dataset('./data.txt', tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='./my_train_bot',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=modle,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model("./my_train_bot")
    tokenizer.save_pretrained("./my_train_bot")

    # while True:
    #     prompt = input("You: ")
    #     if prompt.lower() in ["exit", "quit"]:
    #         break
    #     inputs = tokenizer.encode(prompt, return_tensors="pt")
    #     outputs = modle.generate(inputs, max_length=100, do_sample=True)
    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print("Bot:", response)


if __name__ == "__main__":
    main()