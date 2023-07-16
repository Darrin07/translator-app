from transformers import *
# importing transformers

src = "en"
dst = "de"
# source & destination languages

task_name = f"translation_{src}_to_{dst}"
model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"

translator = pipeline(task_name, model=model_name, tokenizer=model_name)
# initialize the pipeline using model and tokenizer as arguments.

print(translator("You're a genius.")[0]["translation_text"])

article = """
Albert Einstein ( 14 March 1879 – 18 April 1955) was a German-born theoretical 
physicist, widely acknowledged to be one of the greatest physicists of all time. 
Einstein is best known for developing the theory of relativity, but he also made 
important contributions to the development of the theory of quantum mechanics. 
Relativity and quantum mechanics are together the two pillars of modern physics. 
His mass–energy equivalence formula E = mc2, which arises from relativity theory, 
has been dubbed "the world's most famous equation". 
His work is also known for its influence on the philosophy of science.
He received the 1921 Nobel Prize in Physics "for his services to theoretical 
physics, and especially for his discovery of the law of the photoelectric 
effect", a pivotal step in the development of quantum theory. 
His intellectual achievements and originality resulted in "Einstein" becoming 
synonymous with "genius"
"""
print(translator(article)[0]["translation_text"])

def get_model_and_tokenizer(src_lang, dst_lang):
    """
    With the source and destination laguages, return the appropriate model and
    tokenizer
    :param src_lang: original source language
    :param dst_lang: destination language that source is translated to.
    :return: model, tokenizer
    """
    model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"
    # construct model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# source & destination languages
src = "en"
dst = "zh"

model, tokenizer = get_model_and_tokenizer(src, dst)

prompt = tokenizer.encode(article, return_tensors="pt", max_length=512, truncation=True)
# encode text into tensor using tokenizer
print(prompt)

greedy_outputs = model.generate(prompt)
# generate translation output using greedy search
print(tokenizer.decode(greedy_outputs[0], skip_special_tokens=True))

beam_outputs = model.generate(prompt, num_beams=3)
# generate translation outpput using beam search
print(tokenizer.decode(beam_outputs[0], skip_special_tokens=True))

text = "Hello World!"
inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
beam_outputs = model.generate(
    inputs,
    num_beams=5,
    num_return_sequences=5,
    early_stopping=True,
)
# this will use 5 beams and return 5 sequences to compare translation.

for i, beam_output in enumerate(beam_outputs):
  # use for loop to print each version of translation and separate them with "=".
  print(tokenizer.decode(beam_output, skip_special_tokens=True))
  print("="*50)