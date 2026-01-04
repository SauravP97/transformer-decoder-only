from tokenizer import Tokenizer
from train import Trainer

tokenizer = Tokenizer(dataset_path="./dataset/shakespear-text.txt")
dataset = tokenizer.get_dataset()

trainer = Trainer(
    batch_size=16,
    block_size=32,
    train_test_split=0.9,
    data=tokenizer.encode(dataset),
    embedding_dimension=64,
    vocab_size=tokenizer.vocab_size,
    n_head=4,
    n_layer=4,
    max_iterations=5000,
    learning_rate=1e-3,
    eval_interval=500,
)

loss = trainer.execute_training_loop()
print(f"Final Loss after training: {loss}")

predicted_tokens = trainer.generate()
predicted_text = tokenizer.decode(predicted_tokens, stringify=True)

print("\nPredictions:")
print(predicted_text)
