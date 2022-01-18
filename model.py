import numpy as np
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
#
# print(model.summary())
# print(tokenizer.tokenize("전율을 일으키는 영화. 다시 보고싶은 영화"))
# print(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))

max_seq_len = 128

input_id = tokenizer.encode(
    "전율을 일으키는 영화. 다시 보고싶은 영화",
    max_length=max_seq_len,
    pad_to_max_length=True
)

padding_count = input_id.count(tokenizer.pad_token_id)
attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
token_type_id = [0] * max_seq_len

input_ids = np.array([input_id])
attention_masks = np.array([attention_mask])
token_type_ids = np.array([token_type_id])

encoded_input = [input_ids, attention_masks, token_type_ids]
a, b = model.predict(encoded_input)


print(a[0])
# print(a[0].shape)
print("=" * 100)
print(b[0])
# print(b[0].shape)
