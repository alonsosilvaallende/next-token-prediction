import solara
import random
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
text1 = solara.reactive("Alan Turing theorized that computers would one day become")
@solara.component
def Page():
  with solara.Card(margin=0):
    solara.Markdown("#Next token prediction visualization")
    solara.Markdown("I built this tool to help me understand autoregressive language models. For any given text, it gives the top 10 candidates to be the next token with their respective probabilities. The language model I'm using is the smallest version of GPT-2, with 124M parameters.")
    def on_action_cell(column, row_index):
      text1.value += tokenizer.decode(top_10.indices[0][row_index])
    cell_actions = [solara.CellAction(icon="mdi-thumb-up", name="Select", on_click=on_action_cell)]
    solara.InputText("Enter text:", value=text1, continuous_update=True)
    if text1.value != "":
      tokens = tokenizer.encode(text1.value, return_tensors="pt")
      spans1 = ""
      spans2 = ""
      for i, token in enumerate(tokens[0]):
        random.seed(i)
        random_color = ''.join([random.choice('0123456789ABCDEF') for k in range(6)])
        spans1 += " " + f"<span style='font-family: helvetica; color: #{random_color}'>{token}</span>"
        spans2 += " " + f"""<span style="
            padding: 6px;
            border-right: 3px solid white;
            line-height: 3em;
            font-family: courier;
            background-color: #{random_color};
            color: white;
            position: relative;
          "><span style="
          position: absolute;
          top: 5.5ch;
          line-height: 1em;
          left: -0.5px;
          font-size: 0.45em"> {token}</span>{tokenizer.decode([token])}</span>"""
      solara.Markdown(f"{spans1}")
      solara.Markdown(f'{spans2}')
      outputs = model.generate(tokens, max_new_tokens=1, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
      scores = F.softmax(outputs.scores[0], dim=-1)
      top_10 = torch.topk(scores, 10)
      df = pd.DataFrame()
      df["probs"] = top_10.values[0]
      df["probs"] = [f"{value:.2%}" for value in df["probs"].values]
      df["next token ID"] = [top_10.indices[0][i].numpy() for i in range(10)]
      df["next token"] = [tokenizer.decode(top_10.indices[0][i]) for i in range(10)]
      solara.Markdown("###Predictions")
      solara.DataFrame(df, items_per_page=10, cell_actions=cell_actions)
Page()
