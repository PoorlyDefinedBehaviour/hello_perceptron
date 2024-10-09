use rand::Rng;

fn perceptron(inputs: &[f32], weights: &[f32], threshold: f32) -> f32 {
    let sum: f32 = inputs
        .iter()
        .zip(weights)
        .map(|(input, weight)| input * weight)
        .sum();

    if sum >= threshold {
        1.0
    } else {
        0.0
    }
}

// Setting weights and threshold by hand.
fn is_non_negative(x: i32) -> bool {
    perceptron(&[x as f32], &[1.0], 0.0) == 1.0
}

// Setting weights and threshold by hand.
fn not(x: bool) -> bool {
    // 1 * -1 >= 0 -> false
    // 0 * -1 >= 0 -> true
    perceptron(&[if x { 1.0 } else { 0.0 }], &[-1.0], 0.0) == 1.0
}

fn train_perceptron(
    data: &[(Vec<f32>, f32)],
    learning_rate: f32,
    max_iters: usize,
) -> (Vec<f32>, f32) {
    let num_inputs = data[0].0.len();

    let mut weights: Vec<f32> = (0..num_inputs).map(|_| rand::thread_rng().gen()).collect();
    let mut threshold: f32 = rand::thread_rng().gen();

    for _ in 0..max_iters {
        let mut num_errors = 0;

        for (inputs, expected_output) in data {
            let output = perceptron(inputs, &weights, threshold);
            let error = expected_output - output;

            if error != 0.0 {
                num_errors += 1;
                for i in 0..num_inputs {
                    weights[i] += learning_rate * error * inputs[i];
                }
                threshold -= learning_rate * error;
            }
        }

        if num_errors == 0 {
            break;
        }
    }

    (weights, threshold)
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;

    use super::*;

    quickcheck! {
      fn test_is_non_negative(x: i32) -> bool {
          (x >= 0) == is_non_negative(x)
      }

      #[allow(clippy::bool_comparison)]
      fn test_not(x: bool) -> bool {
        (!x) == not(x)
      }
    }

    #[test]
    fn test_perceptron_to_learn_the_and_function() {
        let training_data: Vec<(Vec<f32>, f32)> = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.0),
            (vec![1.0, 0.0], 0.0),
            (vec![1.0, 1.0], 1.0),
        ];

        let (weights, threshold) = train_perceptron(&training_data, 0.1, 10_000);

        assert_eq!(0.0, perceptron(&[0.0, 0.0], &weights, threshold));
        assert_eq!(0.0, perceptron(&[1.0, 0.0], &weights, threshold));
        assert_eq!(0.0, perceptron(&[0.0, 1.0], &weights, threshold));
        assert_eq!(1.0, perceptron(&[1.0, 1.0], &weights, threshold));
    }

    #[test]
    fn test_perceptron_to_learn_the_or_function() {
        let training_data: Vec<(Vec<f32>, f32)> = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 1.0),
        ];

        let (weights, threshold) = train_perceptron(&training_data, 0.1, 10_000);

        assert_eq!(0.0, perceptron(&[0.0, 0.0], &weights, threshold));
        assert_eq!(1.0, perceptron(&[1.0, 0.0], &weights, threshold));
        assert_eq!(1.0, perceptron(&[0.0, 1.0], &weights, threshold));
        assert_eq!(1.0, perceptron(&[1.0, 1.0], &weights, threshold));
    }
}
