import click
import os
import sys

@click.group()
def cli():
    """ecDNA analysis tool (otk)"""
    pass

@cli.command()
@click.option('--config', '-c', default='configs/model_config.yml', help='Configuration file path')
@click.option('--output', '-o', default='models/', help='Output directory for trained models')
@click.option('--gpu', '-g', type=int, default=0, help='GPU device ID to use')
def train(config, output, gpu):
    """Train the ecDNA prediction model"""
    from otk.train.trainer import train_model
    train_model(config, output, gpu)

@cli.command()
@click.option('--model', '-m', required=True, help='Path to trained model')
@click.option('--input', '-i', required=True, help='Input data file path')
@click.option('--output', '-o', default='predictions/', help='Output directory for predictions')
@click.option('--gpu', '-g', type=int, default=-1, help='GPU device ID to use (-1 for CPU)')
def predict(model, input, output, gpu):
    """Predict ecDNA status using trained model"""
    from otk.predict.predictor import predict
    predict(model, input, output, gpu)

if __name__ == '__main__':
    cli()
