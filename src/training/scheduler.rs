/// Learning rate schedulers
#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    /// Constant learning rate
    Constant { lr: f64 },
    /// Step decay: multiply by gamma every step_size epochs
    Step { lr: f64, step_size: usize, gamma: f64 },
    /// Exponential decay: lr * gamma^epoch
    Exponential { lr: f64, gamma: f64 },
    /// Cosine annealing
    Cosine { lr: f64, t_max: usize, eta_min: f64 },
    /// Cosine annealing with warm restarts
    CosineWithRestarts { lr: f64, t_0: usize, t_mult: usize, eta_min: f64 },
    /// Reduce on plateau
    ReduceOnPlateau {
        lr: f64,
        factor: f64,
        patience: usize,
        min_lr: f64,
        cooldown: usize,
        cooldown_counter: usize,
        num_bad_epochs: usize,
    },
}

impl LearningRateScheduler {
    /// Create constant scheduler
    pub fn constant(lr: f64) -> Self {
        Self::Constant { lr }
    }
    
    /// Create step scheduler
    pub fn step(lr: f64, step_size: usize, gamma: f64) -> Self {
        Self::Step { lr, step_size, gamma }
    }
    
    /// Create exponential scheduler
    pub fn exponential(lr: f64, gamma: f64) -> Self {
        Self::Exponential { lr, gamma }
    }
    
    /// Create cosine annealing scheduler
    pub fn cosine(lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self::Cosine { lr, t_max, eta_min }
    }
    
    /// Create cosine with restarts scheduler
    pub fn cosine_with_restarts(lr: f64, t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        Self::CosineWithRestarts { lr, t_0, t_mult, eta_min }
    }
    
    /// Create reduce on plateau scheduler
    pub fn reduce_on_plateau(lr: f64, factor: f64, patience: usize, min_lr: f64) -> Self {
        Self::ReduceOnPlateau {
            lr,
            factor,
            patience,
            min_lr,
            cooldown: 0,
            cooldown_counter: 0,
            num_bad_epochs: 0,
        }
    }
    
    /// Get current learning rate
    pub fn get_lr(&self, epoch: usize) -> f64 {
        match *self {
            Self::Constant { lr } => lr,
            Self::Step { lr, step_size, gamma } => {
                let num_steps = epoch / step_size;
                lr * gamma.powi(num_steps as i32)
            }
            Self::Exponential { lr, gamma } => {
                lr * gamma.powi(epoch as i32)
            }
            Self::Cosine { lr, t_max, eta_min } => {
                if epoch >= t_max {
                    eta_min
                } else {
                    let progress = epoch as f64 / t_max as f64;
                    eta_min + (lr - eta_min) * (1.0 + std::f64::consts::PI * progress).cos() / 2.0
                }
            }
            Self::CosineWithRestarts { lr, t_0, t_mult, eta_min } => {
                // Find current restart period
                let mut t_cur = epoch;
                let mut t_i = t_0;
                while t_cur >= t_i {
                    t_cur -= t_i;
                    t_i *= t_mult;
                }
                let progress = t_cur as f64 / t_i as f64;
                eta_min + (lr - eta_min) * (1.0 + std::f64::consts::PI * progress).cos() / 2.0
            }
            Self::ReduceOnPlateau { lr, .. } => lr,
        }
    }
    
    /// Step the scheduler (for ReduceOnPlateau)
    pub fn step_with_metric(&mut self, metric: f64) -> f64 {
        match self {
            Self::ReduceOnPlateau {
                lr,
                factor,
                patience,
                min_lr,
                cooldown,
                cooldown_counter,
                num_bad_epochs,
            } => {
                // Decrement cooldown counter
                if *cooldown_counter > 0 {
                    *cooldown_counter -= 1;
                    return *lr;
                }
                
                // Check if metric improved (assuming higher is better)
                // This is a simplified version - in practice, you'd track best metric
                *num_bad_epochs += 1;
                
                if *num_bad_epochs > *patience {
                    let new_lr = (*lr * *factor).max(*min_lr);
                    if new_lr < *lr {
                        *lr = new_lr;
                        *cooldown_counter = *cooldown;
                        *num_bad_epochs = 0;
                    }
                }
                
                *lr
            }
            _ => {
                // For other schedulers, just return current LR
                // Epoch is tracked externally
                self.get_lr(0)
            }
        }
    }
    
    /// Update learning rate (for schedulers that need manual updates)
    pub fn set_lr(&mut self, new_lr: f64) {
        match self {
            Self::Constant { lr } => *lr = new_lr,
            Self::Step { lr, .. } => *lr = new_lr,
            Self::Exponential { lr, .. } => *lr = new_lr,
            Self::Cosine { lr, .. } => *lr = new_lr,
            Self::CosineWithRestarts { lr, .. } => *lr = new_lr,
            Self::ReduceOnPlateau { lr, .. } => *lr = new_lr,
        }
    }
}

impl Default for LearningRateScheduler {
    fn default() -> Self {
        Self::constant(0.001)
    }
}

/// Warmup scheduler that gradually increases LR from 0 to target
pub struct WarmupScheduler {
    /// Target learning rate
    target_lr: f64,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Current step
    current_step: usize,
    /// Base scheduler to use after warmup
    base_scheduler: LearningRateScheduler,
}

impl WarmupScheduler {
    /// Create new warmup scheduler
    pub fn new(target_lr: f64, warmup_steps: usize, base_scheduler: LearningRateScheduler) -> Self {
        Self {
            target_lr,
            warmup_steps,
            current_step: 0,
            base_scheduler,
        }
    }
    
    /// Get learning rate for current step
    pub fn get_lr(&mut self, epoch: usize) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let warmup_lr = self.target_lr * (self.current_step as f64 / self.warmup_steps as f64);
            self.current_step += 1;
            warmup_lr
        } else {
            // Use base scheduler
            self.base_scheduler.get_lr(epoch)
        }
    }
    
    /// Step with metric (for ReduceOnPlateau)
    pub fn step_with_metric(&mut self, metric: f64) -> f64 {
        if self.current_step < self.warmup_steps {
            self.get_lr(0)
        } else {
            self.base_scheduler.step_with_metric(metric)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_scheduler() {
        let scheduler = LearningRateScheduler::constant(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(10), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
    }
    
    #[test]
    fn test_step_scheduler() {
        let scheduler = LearningRateScheduler::step(0.001, 10, 0.5);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(9), 0.001);
        assert_eq!(scheduler.get_lr(10), 0.0005);
        assert_eq!(scheduler.get_lr(19), 0.0005);
        assert_eq!(scheduler.get_lr(20), 0.00025);
    }
    
    #[test]
    fn test_exponential_scheduler() {
        let scheduler = LearningRateScheduler::exponential(0.001, 0.9);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert!((scheduler.get_lr(1) - 0.0009).abs() < 1e-10);
        assert!((scheduler.get_lr(10) - 0.001 * 0.9f64.powi(10)).abs() < 1e-15);
    }
    
    #[test]
    fn test_cosine_scheduler() {
        let scheduler = LearningRateScheduler::cosine(0.001, 100, 0.0001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.0001);
        // At half way, should be roughly average
        let mid_lr = scheduler.get_lr(50);
        assert!(mid_lr > 0.0001 && mid_lr < 0.001);
    }
    
    #[test]
    fn test_warmup_scheduler() {
        let base = LearningRateScheduler::constant(0.001);
        let mut warmup = WarmupScheduler::new(0.001, 10, base);
        
        assert!(warmup.get_lr(0) < 0.001);
        assert_eq!(warmup.get_lr(0), 0.0001); // After first call
        
        // After warmup, should use base scheduler
        for _ in 0..10 {
            let _ = warmup.get_lr(0);
        }
        assert_eq!(warmup.get_lr(0), 0.001);
    }
}