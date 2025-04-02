the dspy elo prediction model can be trained using bootstrapfewshot to more accurately predict which of two samples has higher training elo. there are integration tests. there's a csv with data that can be used to test training. the csv actually does contain some text and for each text it has a rating from 1 to 9. there is code that uses those and creates an elo score for each.
- model deepseek/deepseek-chat
- we use the new dspy.LM to connect to the model
- the outputs are compared by a dspy module which decides on the better output
- readme documents how to run
  - training
  - inference on a new sample
  - how to use your own dataset
