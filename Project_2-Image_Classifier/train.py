import argparse
import helper


available_architectures = {'vgg19_bn', 'densenet201', 'resnet152'}

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Model Settings")
    
    #positional arguments:
    parser.add_argument('data_dir', default = 'flowers', help = 'Directory with flower images')
    
    #positional arguments:
    parser.add_argument('-s', '--save_dir', default = 'checkpoint', help='Directory for checkpoint saving')
    parser.add_argument('-a', '--arch', default = 'resnet152', action = 'store', choices = available_architectures, help = 'Available architectures')
    parser.add_argument('-l', '--learning_rate', type = float, default = 0.001, help = 'Learning rate of the model')
    parser.add_argument('-H', '--hidden_units', type = int, default = 1024, help = 'Amount of hidden units')
    parser.add_argument('-e', '--epochs', type = int, default = 6, help = 'Number of epochs')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU if available')
    
    # Parse args
    args = parser.parse_args()
    return args



def main():
    
    # Get input arguments
    args = arg_parser()
    
    # Process and load the data/images
    image_datasets, dataloaders = helper.process_and_load_data(args.data_dir)
    print("The train, test & validation data has been loaded.".format(key))
    
    # Load the model
    model, optimizer, criterion = helper.build_model(args.arch, args.hidden_units, args.learning_rate)
    print("Model, optimizer & criterion have been loaded.")
    
    # Check if GPU is available
    device = helper.check_gpu(args.gpu)
    print('Using {} for computation.'.format(device))
    
    # Train and validate the model
    helper.train_and_validate_model(model, optimizer, criterion, dataloaders, device, args.epochs, print_every = 32)
    print("Training has been completed.")
    
    # Test the model
    helper.test_model(model, optimizer, criterion, dataloaders, device)
    print("Testing has been completed.")
    
    # Save the checkpoint
    helper.save_checkpoint(args.arch, model, args.epochs, args.hidden_units, args.learning_rate, image_datasets, args.save_dir)
    print("Model's checkpoint has been saved.")
    
if __name__ == '__main__':
    main()