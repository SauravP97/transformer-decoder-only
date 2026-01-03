from tokenizer import Tokenizer
from train import Trainer

tokenizer = Tokenizer(dataset_path="./dataset/shakespear-text.txt")
dataset = tokenizer.get_dataset()

# print(dataset[:200])

# encoded_value = tokenizer.encode("Hey How are You")
# print(encoded_value)
# print(tokenizer.decode(encoded_value))
# print(tokenizer.decode(encoded_value, True))

trainer = Trainer(
    batch_size=3, block_size=8, train_test_split=0.9, data=tokenizer.encode(dataset)
)

xb, yb = trainer.get_batch_of_train_or_test_split(split="Train")
print(f"Input: {xb}")
print(f"Target: {yb}")

print("Decoded the batch")

print(f"Input : {[tokenizer.decode(x, stringify=True) for x in xb.numpy()]}")
print(f"Target: {[tokenizer.decode(y, stringify=True) for y in yb.numpy()]}")
