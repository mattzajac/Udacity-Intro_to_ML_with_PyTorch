import argparse
import helper


def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Model Settings")
    
    #positional arguments:
    parser.add_argument('-i', '--img_path', required=True, default = 'flowers/test/2/image_05100.jpg', help = 'Location of the image to be classified')
    parser.add_argument('-c', '--checkpoint_path', required=True, default = 'checkpoint/resnet152_checkpoint.pth', help = 'Location of the model checkpoint to be used')
    
    #positional arguments:
    parser.add_argument('-n', '--category_names', default = 'cat_to_name.json', help = 'Category names')
    parser.add_argument('-t', '--top_k', type = int, default = 3, help = 'Return the top K highest probability classes')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU if available')
    
    # Parse args
    args = parser.parse_args()
    return args


def main():
    
    # Get input arguments
    args = arg_parser()
    
    # Check if GPU is available
    device = helper.check_gpu(args.gpu)
    print('Using {} for computation.'.format(device))
    
    # Load the model from checkpoint
    model = helper.load_checkpoint(args.checkpoint_path)
    print("Model has been loaded from the checkpoint.")
    print("You will get predictions in a bit...")
    
    # Get predictions for the chosen image
    top_probs, top_labels, top_flowers = helper.predict(model, args.img_path, args.top_k)
    
    # Print top n probabilities
    helper.print_probability(top_flowers, top_probs)
        
if __name__ == '__main__':
    main()