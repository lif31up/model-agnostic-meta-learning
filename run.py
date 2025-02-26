import argparse
import src.train as train
import src.eval as eval
import torchvision as tv

def main():
  # eval(default)
  parser = argparse.ArgumentParser(description="Few-shot learning using MAML")
  parser.add_argument("--model", type=str, help="path of your model")

  # train
  subparser = parser.add_subparsers(title="subcommands", dest="subcommand")
  parser_train = subparser.add_parser("train", help="train your model")
  parser_train.add_argument("--path", type=str, help="path to your dataset")
  parser_train.add_argument("--save_to", type=str, help="path to save your model")
  parser_train.add_argument("--n_way", type=int, help="number of classes per episode")
  parser_train.add_argument("--k_shot", type=int, help="number of support samples per class")
  parser_train.add_argument("--n_query", type=int, help="number of query samples per class")
  parser_train.add_argument("--iters", type=int, help="how much iteration your model does for an episode")
  parser_train.add_argument("--epochs", type=int, help="how much epochs your model does for training")
  parser_train.set_defaults(func=lambda kwargs: train.main(
    path=kwargs.path,
    save_to=kwargs.save_to,
    n_way=kwargs.n_way,
    k_shot=kwargs.k_shot,
    n_query=kwargs.n_query,
    epochs=kwargs.epochs,
    iters=kwargs.iters)
  ) # parser_train.set_defaults()

  # download dataset
  parser_download = subparser.add_parser("download", help="download dataset")
  parser_download.add_argument("--path", type=str, help="path to download dataset")
  parser_download.set_defaults(func=lambda kwargs: tv.datasets.Omniglot(root=kwargs.path, background=True, download=True))

  args = parser.parse_args()
  if hasattr(args, 'func'): args.func(args)
  elif args.model: eval.main(model=args.model)
  else: print("invalid argument. exiting program.")
# main():

if __name__ == "__main__": main()