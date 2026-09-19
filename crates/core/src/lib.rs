//! Moteur de jeu 2048 (chapitre 2). Placeholder pour le socle du chapitre 1.

pub fn hello() -> String {
    "hello from g2048-core".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_expected_string() {
        assert_eq!(hello(), "hello from g2048-core");
    }
}
