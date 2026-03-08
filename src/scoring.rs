use crate::accessor::NodeView;
use crate::types::{DynamicContext, NodeResult};

/// Compute the dynamic score for a node given a traversal context.
///
/// Uses the **multiplicative noise** formula (spec Risk #17):
/// ```text
/// raw = (w_base * base_weight) + (w_semantic * semantic_boost) + (w_override * weight_override)
/// W_total = raw * (1.0 - effective_noise)
/// ```
///
/// This ensures that even heavily boosted nodes are suppressed when `noise_penalty` is high.
pub fn compute_score(node: &NodeView, ctx: &DynamicContext) -> NodeResult {
    let base_component = ctx.w_base * node.base_weight;

    let semantic_boost = ctx
        .semantic_boosts
        .get(&node.name)
        .copied()
        .unwrap_or(0.0);
    let semantic_component = ctx.w_semantic * semantic_boost;

    let weight_override = ctx
        .weight_overrides
        .get(&node.name)
        .copied()
        .unwrap_or(0.0);
    let override_component = ctx.w_override * weight_override;

    // Domain-aware noise: if node metadata contains a tag in ctx.noise_tags, force high penalty
    let mut effective_noise = (ctx.w_noise * node.noise_penalty).clamp(0.0, 1.0);
    if !ctx.noise_tags.is_empty() {
        if let Some(domain) = node.metadata.get("domain") {
            if ctx.noise_tags.contains(domain) {
                effective_noise = effective_noise.max(0.9);
            }
        }
    }

    let raw = base_component + semantic_component + override_component;
    let weight = raw * (1.0 - effective_noise);

    NodeResult {
        name: node.name.clone(),
        kind: node.kind.clone(),
        weight,
        base_component,
        semantic_component,
        noise_component: effective_noise,
        override_component,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_node(name: &str, base_weight: f32, noise: f32) -> NodeView {
        NodeView {
            name: name.to_string(),
            kind: "model".to_string(),
            metadata: HashMap::new(),
            base_weight,
            noise_penalty: noise,
            pagerank_weight: 0.0,
        }
    }

    fn make_node_with_domain(name: &str, base_weight: f32, noise: f32, domain: &str) -> NodeView {
        let mut metadata = HashMap::new();
        metadata.insert("domain".to_string(), domain.to_string());
        NodeView {
            name: name.to_string(),
            kind: "model".to_string(),
            metadata,
            base_weight,
            noise_penalty: noise,
            pagerank_weight: 0.0,
        }
    }

    #[test]
    fn test_basic_score_no_context() {
        let node = make_node("sale.order", 0.7, 0.0);
        let ctx = DynamicContext::default();
        let result = compute_score(&node, &ctx);
        assert!((result.weight - 0.7).abs() < 0.001);
        assert!((result.base_component - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_semantic_boost_raises_score() {
        let node = make_node("stock.picking", 0.5, 0.0);
        let mut ctx = DynamicContext::default();
        ctx.semantic_boosts
            .insert("stock.picking".to_string(), 2.0);
        let result = compute_score(&node, &ctx);
        assert!((result.weight - 3.5).abs() < 0.001);
        assert!((result.semantic_component - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_noise_kills_node() {
        let node = make_node("create_uid", 0.8, 0.9);
        let ctx = DynamicContext::default();
        let result = compute_score(&node, &ctx);
        assert!(result.weight < 0.1, "noise=0.9 should kill node: {}", result.weight);
    }

    #[test]
    fn test_multiplicative_noise_beats_boost() {
        let node = make_node("create_uid", 0.8, 0.9);
        let mut ctx = DynamicContext::default();
        ctx.semantic_boosts
            .insert("create_uid".to_string(), 5.0);
        let result = compute_score(&node, &ctx);
        assert!(result.weight < 1.0, "Boosted noisy node should still score low: {}", result.weight);
    }

    #[test]
    fn test_noise_tags_domain_penalty() {
        let node = make_node_with_domain("ir.cron", 0.5, 0.1, "technical");
        let mut ctx = DynamicContext::default();
        ctx.noise_tags.insert("technical".to_string());
        let result = compute_score(&node, &ctx);
        assert!(result.weight < 0.1, "Domain-tagged node should be penalized: {}", result.weight);
        assert!((result.noise_component - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_weight_override() {
        let node = make_node("sale.order", 0.3, 0.0);
        let mut ctx = DynamicContext::default();
        ctx.weight_overrides.insert("sale.order".to_string(), 0.5);
        let result = compute_score(&node, &ctx);
        assert!((result.weight - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_explain_score_components() {
        let node = make_node("sale.order", 0.6, 0.2);
        let mut ctx = DynamicContext::default();
        ctx.semantic_boosts.insert("sale.order".to_string(), 1.0);
        ctx.weight_overrides.insert("sale.order".to_string(), 0.3);

        let result = compute_score(&node, &ctx);
        let explained = result.explain_score();

        assert!((explained["base"] - 0.6).abs() < 0.001);
        assert!((explained["semantic"] - 1.5).abs() < 0.001);
        assert!((explained["override"] - 0.3).abs() < 0.001);
        assert!((explained["noise"] - 0.2).abs() < 0.001);
        assert!((explained["total"] - 1.92).abs() < 0.001);
    }
}
