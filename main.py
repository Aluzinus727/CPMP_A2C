from a2c import A2CTrainer
from config import Config


if __name__ == "__main__":
    config = Config(
        observation_size=56,
        action_space_size=42,
        actor_layers = [32],
        critic_layers = [64, 64],
        lr=1e-3,
        encoding_size=16,
        batch_size=64,
        nb_rollout_steps=16,
        normalize_advantage=True,
        max_grad_norm = 0.5,
        vf_coef=0.5,
        ent_coef=1e-3
    )
    trainer = A2CTrainer(config)
    trainer.train(greedy=True)
    trainer.reset_variables()
    trainer.train()