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
