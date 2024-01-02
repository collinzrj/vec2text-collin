from transformers import AutoTokenizer


prompt = 'Registration 25 june 1859 21:15 Make this lower case'
entity = '25 june 1859'
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
print(roberta_tokenizer.decode([1]))
exit()
enc = roberta_tokenizer(prompt)
tokens = roberta_tokenizer.encode(prompt)
start_char_idx = prompt.find(entity)
end_char_idx = start_char_idx + len(entity) - 1
start_token = enc.char_to_token(start_char_idx)
end_token = enc.char_to_token(end_char_idx)
print(start_char_idx)
print(end_char_idx)
print(start_token)
print(end_token)
# end_token should be included
print(roberta_tokenizer.decode(tokens[start_token:end_token+1]))

# ## TODO: understand what this means
# print([roberta_tokenizer.decode(token) for token in tokens])
# print(enc.word_ids())
# words = prompt.split(' ')
# for idx, word_id in enumerate(enc.word_ids()):
#     if word_id is not None:
#         span = enc.word_to_chars(word_id)
#         print(prompt[span.start:span.end])
#         enc.char_to_token()
#         # print(tokens[idx], roberta_tokenizer.decode(tokens[idx]), words[word_id])
# # entity_tokens = roberta_tokenizer.encode(entity)
