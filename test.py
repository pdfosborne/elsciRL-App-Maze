from elsciRL.experiments.standard import Experiment
from environment.engine import Engine
from adapters.language import Adapter as LanguageAdapter
import json
import os

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    # Load configurations
    experiment_config = load_config('configs/config.json')
    problem_config = load_config('configs/config_local.json')
    
    # Create experiment directory
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize experiment
    experiment = Experiment(
        Config=experiment_config,
        ProblemConfig=problem_config,
        Engine=Engine,
        Adapters={"language": LanguageAdapter},
        save_dir=save_dir,
        show_figures='Y',
        window_size=10.0,
        training_render=True,
        training_render_save_dir=os.path.join(save_dir, "renders")
    )
    
    # Run experiment
    experiment.train()
    experiment.test()

if __name__ == "__main__":
    main()



