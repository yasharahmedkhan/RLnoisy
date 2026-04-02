// -*- coding: utf-8 -*-
/*
(C) Copyright 2025 IBM. All Rights Reserved.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
*/

use pyo3::prelude::*;

use rand::distributions::{Distribution, Uniform};

use twisterl::rl::env::Env;
use twisterl::python_interface::env::{PyBaseEnv, get_env_ref, get_env_mut};
use std::collections::HashMap;


use crate::envs::common::Gate;

/// Type alias for a noise model: maps (q1, q2) edge pairs to their noise penalty (negative float).
/// Edges not in this map fall back to `default_noise_rate`.
pub type NoiseModel = HashMap<(usize, usize), f32>;

// Define some internal representation
#[derive(Clone)]
pub struct LFState {
    pub data: Vec<bool>,
    pub map: HashMap<(usize, usize), f32>,
    size: usize,
}

// Here are some functions to manipulate the internal representation
impl LFState {
    // Constructor to create a new LinearFunction
    fn new(size: usize) -> Self {
        let lf = LFState {
            data: vec![false; size * size],
            size,
            map: HashMap::new(),
        };
        let mut lf = lf;
        for i in 0..size {
            lf.set(i, i, true);
        }
        lf
    }

    // Method to set a value in the LinearFunction
    fn set(&mut self, row: usize, column: usize, value: bool) {
        let index = self.index(row, column);
        self.data[index] = value;
    }

    // Method to get a value from the LinearFunction
    fn get(&self, row: usize, column: usize) -> bool {
        let index = self.index(row, column);
        self.data[index]
    }

    fn addNoise(&mut self, row: usize, column: usize, value: f32) {
        *self.map.entry((row, column)).or_insert(0.0) += value;
    }

    /// Sets (overwrites) noise value for a given edge
    fn setNoise(&mut self, row: usize, column: usize, value: f32) {
        self.map.insert((row, column), value);
    }

    /// Gets the noise value for a given edge, returning 0.0 if not present
    fn getNoise(&self, row: usize, column: usize) -> f32 {
        *self.map.get(&(row, column)).unwrap_or(&0.0)
    }

    ///clears all accumulated noise
    fn clearNoise(&mut self) {
        self.map.clear();
    }

    ///returns total accumulated noise across all edges
    fn totalNoise(&self) -> f32 {
        self.map.values().sum()
    }

    // Method to perform cx between q1 and q2
    fn cx(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for column in 0..self.size {
            let a_val = self.get(q1, column);
            let b_val = self.get(q2, column);
            self.set(q2, column, a_val ^ b_val);
        }
    }

    fn swap(&mut self, q1: usize, q2: usize) {
        if q1 == q2 {
            return;
        }
        for column in 0..self.size {
            let a_val = self.get(q1, column);
            let b_val = self.get(q2, column);
            self.set(q1, column, b_val);
            self.set(q2, column, a_val);
        }
    }

    // Private helper method to calculate the linear index from row and column
    fn index(&self, row: usize, column: usize) -> usize {
        row * self.size + column
    }

    // Check if it is identity
    fn solved(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if ((i == j) && (self.get(i, j) != true)) || ((i != j) && (self.get(i, j) != false)) {
                    return false;
                }
            }
        }
        true
    }
}

// This is the Env definition
#[derive(Clone)]
pub struct LinearFunctionNoisy {
    pub lf: LFState,
    pub depth: usize,
    pub success: bool,

    pub difficulty: usize,
    pub gateset: Vec<Gate>,
    pub depth_slope: usize,
    pub max_depth: usize,
    pub recent_noise: f32,
    pub total_noise: f32,
    pub edge_noise_rates: HashMap<(usize, usize), f32>,
    pub default_noise_rate: f32,
}


impl LinearFunctionNoisy {
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        noise_rates: Vec<(usize, usize, f32)>,
        default_noise_rate: f32,
    ) -> Self {
        let lf = LFState::new(num_qubits);
        let success = lf.solved();

        let mut edge_noise_rates = HashMap::new();
        for (q1, q2, rate) in noise_rates {
            let key = if q1 <= q2 { (q1, q2) } else { (q2, q1) };
            edge_noise_rates.insert(key, rate);
        }

        LinearFunctionNoisy {
            lf, depth: 1, success, difficulty, gateset, depth_slope, max_depth,
            recent_noise: 0.0,
            total_noise: 0.0,
            edge_noise_rates,
            default_noise_rate,
        }
    }

    pub fn solved(&self) -> bool {
        self.lf.solved()
    }

    ///get the noise rate for a specific edge
    fn get_edge_noise_rate(&self, q1: usize, q2: usize) -> f32 {
        let key = if q1 <= q2 { (q1, q2) } else { (q2, q1) };
        *self.edge_noise_rates.get(&key).unwrap_or(&self.default_noise_rate)
    }
}

// This implements the necessary functions for the environment
impl Env for LinearFunctionNoisy {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn num_actions(&self) -> usize {
        self.gateset.len()
    }

    fn obs_shape(&self) -> Vec<usize> {
        vec![self.lf.size, self.lf.size]
    }

    fn set_difficulty(&mut self, difficulty: usize) {
        self.difficulty = difficulty;
    }

    fn get_difficulty(&self) -> usize {
        self.difficulty
    }

    fn set_state(&mut self, state: Vec<i64>) {
        self.lf.data = state.iter().map(|&x| x>0).collect();
        self.depth = self.max_depth;
        self.success = self.solved();
        //clear noise when state is set externally
        self.lf.clearNoise();
        self.total_noise = 0.0;
        self.recent_noise = 0.0;
    }

    fn reset(&mut self) {
        // Create an identity matrix for the initial 'lf' state
        self.lf = LFState::new(self.lf.size);
        self.depth = self.max_depth;
        self.success = self.solved();

        let mut rng = rand::thread_rng();
        let action_range = Uniform::new(0, self.num_actions());

        // Apply random actions based on the difficulty
        for _ in 0..self.difficulty {
            let action = action_range.sample(&mut rng);
            self.step(action);
        }
        self.depth = (self.depth_slope * self.difficulty).min(self.max_depth);
        self.success = self.solved();

        self.lf.clearNoise();
        self.total_noise = 0.0;
        self.recent_noise = 0.0;
    }

    fn step(&mut self, action: usize)  {
        match self.gateset[action] {
            Gate::CX(q1, q2) => {
                let noise = self.get_edge_noise_rate(q1, q2);
                self.lf.cx(q1, q2);
                self.lf.addNoise(q1, q2, noise);
                self.recent_noise = noise;
                self.total_noise += noise;
            }
            Gate::SWAP(q1, q2) => {
                let noise = self.get_edge_noise_rate(q1, q2) * 3.0;
                self.lf.swap(q1, q2);
                self.lf.addNoise(q1, q2, noise);
                self.recent_noise = noise;
                self.total_noise += noise;
            }
            _ => {}
        }
        self.depth = self.depth.saturating_sub(1); 
        self.success = self.solved();
    }

    fn masks(&self) -> Vec<bool> {
        vec![!self.success; self.num_actions()]
    }

    fn is_final(&self) -> bool {
        self.depth == 0 || self.success
    }

    fn success(&self) -> bool { self.success }

    fn reward(&self) -> f32 {
        if self.success {
            (1.0 + self.total_noise).max(0.1)
        } else {
            if self.depth == 0 {
                -0.5 + self.recent_noise
            } else {
                (-0.5 + self.recent_noise) / self.max_depth as f32
            }
        }
    }

    fn observe(&self,) -> Vec<usize> {
        self.lf.data.iter()
        .enumerate() // Iterate over the Vec with indices
        .filter_map(|(index, &value)| if value { Some(index) } else { None }) // Collect indices where the value is true
        .collect()
    }
}


#[pyclass(name="LinearFunctionNoisyEnv", extends=PyBaseEnv)]
pub struct PyLinearFunctionNoisyEnv;

#[pymethods]
impl PyLinearFunctionNoisyEnv {
    #[new]
    pub fn new(
        num_qubits: usize,
        difficulty: usize,
        gateset: Vec<Gate>,
        depth_slope: usize,
        max_depth: usize,
        noise_rates: Vec<(usize, usize, f32)>,
        default_noise_rate: f32,
    ) -> (Self, PyBaseEnv) {
        let env = LinearFunctionNoisy::new(
            num_qubits, difficulty, gateset, depth_slope, max_depth,
            noise_rates, default_noise_rate,
        );
        let env = Box::new(env);
        (PyLinearFunctionNoisyEnv, PyBaseEnv { env })
    }
}
