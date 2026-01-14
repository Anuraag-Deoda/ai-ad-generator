"""
Hooks Database Module
Provides trending hooks templates and A/B hook generation for ad scripts.
"""

import random
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class HookTemplate:
    """A hook template with placeholders."""
    template: str
    strategy: str
    energy: str  # high, medium, low
    platforms: List[str]  # Best platforms for this hook style
    industries: List[str]  # Best industries


# Trending hooks database organized by strategy
TRENDING_HOOKS: Dict[str, List[HookTemplate]] = {
    "question": [
        HookTemplate(
            "What if I told you {benefit}?",
            "question", "medium",
            ["tiktok", "instagram", "youtube"],
            ["ecommerce", "saas", "services"]
        ),
        HookTemplate(
            "Why is everyone talking about {product}?",
            "question", "high",
            ["tiktok", "instagram"],
            ["ecommerce", "beauty", "fashion"]
        ),
        HookTemplate(
            "Have you tried {solution} yet?",
            "question", "medium",
            ["instagram", "facebook"],
            ["ecommerce", "health", "beauty"]
        ),
        HookTemplate(
            "What's stopping you from {desired_outcome}?",
            "question", "medium",
            ["linkedin", "youtube"],
            ["saas", "services", "coaching"]
        ),
        HookTemplate(
            "Ever wondered how {audience} get {result}?",
            "question", "medium",
            ["tiktok", "youtube"],
            ["fitness", "finance", "education"]
        ),
        HookTemplate(
            "Can {product} really {bold_claim}?",
            "question", "high",
            ["tiktok", "instagram"],
            ["ecommerce", "health", "beauty"]
        ),
    ],

    "problem_solution": [
        HookTemplate(
            "Struggling with {pain_point}? Here's the fix.",
            "problem_solution", "medium",
            ["tiktok", "instagram", "facebook"],
            ["ecommerce", "saas", "services"]
        ),
        HookTemplate(
            "Stop {bad_behavior}. Start {good_behavior}.",
            "problem_solution", "high",
            ["tiktok", "instagram"],
            ["health", "productivity", "fitness"]
        ),
        HookTemplate(
            "This {product_type} changed everything.",
            "problem_solution", "high",
            ["tiktok", "instagram", "youtube"],
            ["ecommerce", "tech", "lifestyle"]
        ),
        HookTemplate(
            "{pain_point} is ruining your {aspect}. Until now.",
            "problem_solution", "medium",
            ["facebook", "linkedin"],
            ["saas", "services", "health"]
        ),
        HookTemplate(
            "I was {negative_state} until I found this.",
            "problem_solution", "high",
            ["tiktok", "instagram"],
            ["health", "beauty", "lifestyle"]
        ),
    ],

    "social_proof": [
        HookTemplate(
            "Over {number} people swear by this.",
            "social_proof", "high",
            ["tiktok", "instagram", "facebook"],
            ["ecommerce", "health", "beauty"]
        ),
        HookTemplate(
            "The {product_type} that broke the internet.",
            "social_proof", "high",
            ["tiktok", "instagram"],
            ["ecommerce", "tech", "fashion"]
        ),
        HookTemplate(
            "Why {demographic} can't stop buying this.",
            "social_proof", "high",
            ["tiktok", "instagram", "youtube"],
            ["ecommerce", "beauty", "lifestyle"]
        ),
        HookTemplate(
            "Doctors/Experts recommend this {product_type}.",
            "social_proof", "medium",
            ["facebook", "youtube"],
            ["health", "wellness", "beauty"]
        ),
        HookTemplate(
            "#1 {product_type} in {category}.",
            "social_proof", "medium",
            ["instagram", "facebook", "linkedin"],
            ["ecommerce", "saas", "services"]
        ),
        HookTemplate(
            "Join {number}+ happy customers.",
            "social_proof", "medium",
            ["facebook", "linkedin"],
            ["saas", "services", "ecommerce"]
        ),
    ],

    "curiosity": [
        HookTemplate(
            "Nobody talks about this {product_type} hack.",
            "curiosity", "high",
            ["tiktok", "instagram", "youtube"],
            ["ecommerce", "lifestyle", "tech"]
        ),
        HookTemplate(
            "The {industry} secret they don't want you to know.",
            "curiosity", "high",
            ["tiktok", "youtube"],
            ["finance", "health", "beauty"]
        ),
        HookTemplate(
            "I tested {product} for {time_period}. Here's what happened.",
            "curiosity", "high",
            ["tiktok", "youtube"],
            ["ecommerce", "health", "tech"]
        ),
        HookTemplate(
            "Wait for it...",
            "curiosity", "high",
            ["tiktok", "instagram"],
            ["ecommerce", "lifestyle", "entertainment"]
        ),
        HookTemplate(
            "You won't believe what this {product_type} can do.",
            "curiosity", "high",
            ["tiktok", "instagram", "facebook"],
            ["ecommerce", "tech", "gadgets"]
        ),
    ],

    "urgency": [
        HookTemplate(
            "Last chance to grab {product}!",
            "urgency", "high",
            ["instagram", "facebook", "tiktok"],
            ["ecommerce", "retail", "fashion"]
        ),
        HookTemplate(
            "This deal ends {time}.",
            "urgency", "high",
            ["instagram", "facebook"],
            ["ecommerce", "services", "saas"]
        ),
        HookTemplate(
            "Only {number} left in stock!",
            "urgency", "high",
            ["instagram", "facebook", "tiktok"],
            ["ecommerce", "retail", "fashion"]
        ),
        HookTemplate(
            "Don't miss out on {benefit}.",
            "urgency", "medium",
            ["facebook", "linkedin", "instagram"],
            ["saas", "services", "ecommerce"]
        ),
        HookTemplate(
            "Limited time: {offer}",
            "urgency", "high",
            ["instagram", "facebook"],
            ["ecommerce", "services", "retail"]
        ),
    ],

    "transformation": [
        HookTemplate(
            "From {before_state} to {after_state} in {time}.",
            "transformation", "high",
            ["tiktok", "instagram", "youtube"],
            ["fitness", "beauty", "lifestyle"]
        ),
        HookTemplate(
            "How I {achieved_result} with just {product}.",
            "transformation", "high",
            ["tiktok", "youtube", "instagram"],
            ["health", "fitness", "productivity"]
        ),
        HookTemplate(
            "The {product_type} that transformed my {aspect}.",
            "transformation", "high",
            ["tiktok", "instagram"],
            ["beauty", "health", "lifestyle"]
        ),
        HookTemplate(
            "Watch this transformation.",
            "transformation", "high",
            ["tiktok", "instagram"],
            ["beauty", "fitness", "home"]
        ),
        HookTemplate(
            "{time_period} ago I couldn't {action}. Now look at me.",
            "transformation", "high",
            ["tiktok", "instagram", "youtube"],
            ["fitness", "health", "education"]
        ),
    ],

    "contrarian": [
        HookTemplate(
            "Forget everything you know about {topic}.",
            "contrarian", "high",
            ["tiktok", "youtube", "linkedin"],
            ["education", "saas", "marketing"]
        ),
        HookTemplate(
            "This is NOT your typical {product_type}.",
            "contrarian", "medium",
            ["tiktok", "instagram"],
            ["ecommerce", "tech", "lifestyle"]
        ),
        HookTemplate(
            "Why {common_advice} is wrong.",
            "contrarian", "high",
            ["youtube", "linkedin", "tiktok"],
            ["education", "finance", "health"]
        ),
        HookTemplate(
            "Unpopular opinion: {bold_statement}",
            "contrarian", "high",
            ["tiktok", "twitter", "linkedin"],
            ["tech", "lifestyle", "business"]
        ),
    ],

    "benefit_focused": [
        HookTemplate(
            "Get {benefit} without {common_drawback}.",
            "benefit_focused", "medium",
            ["facebook", "instagram", "linkedin"],
            ["saas", "services", "ecommerce"]
        ),
        HookTemplate(
            "{benefit} in just {time}.",
            "benefit_focused", "high",
            ["tiktok", "instagram", "facebook"],
            ["fitness", "beauty", "productivity"]
        ),
        HookTemplate(
            "Finally, a {product_type} that actually {delivers_on_promise}.",
            "benefit_focused", "medium",
            ["facebook", "youtube", "instagram"],
            ["ecommerce", "health", "tech"]
        ),
        HookTemplate(
            "Save {amount} on {expense}.",
            "benefit_focused", "medium",
            ["facebook", "linkedin"],
            ["finance", "saas", "services"]
        ),
    ],
}


# Industry-specific emotional triggers
EMOTIONAL_TRIGGERS: Dict[str, Dict[str, List[str]]] = {
    "ecommerce": {
        "fomo": ["limited edition", "selling fast", "almost gone", "exclusive"],
        "social_proof": ["bestselling", "5-star rated", "customer favorite", "trending"],
        "value": ["save", "discount", "deal", "free shipping"],
        "quality": ["premium", "handcrafted", "authentic", "original"],
    },
    "saas": {
        "efficiency": ["automate", "streamline", "save time", "boost productivity"],
        "growth": ["scale", "grow", "increase", "optimize"],
        "trust": ["secure", "trusted by", "enterprise-grade", "reliable"],
        "simplicity": ["easy", "simple", "no-code", "one-click"],
    },
    "health": {
        "transformation": ["transform", "change", "improve", "enhance"],
        "natural": ["organic", "natural", "clean", "pure"],
        "results": ["clinically proven", "science-backed", "doctor recommended"],
        "wellness": ["feel better", "energy", "vitality", "balance"],
    },
    "services": {
        "expertise": ["expert", "professional", "certified", "experienced"],
        "results": ["guaranteed", "proven", "results-driven", "effective"],
        "convenience": ["easy", "hassle-free", "quick", "convenient"],
        "trust": ["trusted", "reliable", "recommended", "award-winning"],
    },
}


# Platform-specific copy guidelines
PLATFORM_GUIDELINES: Dict[str, Dict[str, Any]] = {
    "tiktok": {
        "max_hook_words": 8,
        "tone": "casual, trendy, authentic",
        "emoji_usage": "high",
        "cta_style": "short action verbs",
        "best_strategies": ["curiosity", "transformation", "contrarian"],
    },
    "instagram": {
        "max_hook_words": 12,
        "tone": "aspirational, visual-focused",
        "emoji_usage": "medium",
        "cta_style": "lifestyle-oriented",
        "best_strategies": ["social_proof", "transformation", "benefit_focused"],
    },
    "youtube": {
        "max_hook_words": 15,
        "tone": "educational, engaging, detailed",
        "emoji_usage": "low",
        "cta_style": "subscribe, learn more",
        "best_strategies": ["curiosity", "problem_solution", "transformation"],
    },
    "linkedin": {
        "max_hook_words": 15,
        "tone": "professional, data-driven, insightful",
        "emoji_usage": "minimal",
        "cta_style": "business-value focused",
        "best_strategies": ["problem_solution", "social_proof", "benefit_focused"],
    },
    "facebook": {
        "max_hook_words": 12,
        "tone": "friendly, community-focused",
        "emoji_usage": "medium",
        "cta_style": "clear action",
        "best_strategies": ["social_proof", "urgency", "benefit_focused"],
    },
    "twitter": {
        "max_hook_words": 10,
        "tone": "witty, concise, conversational",
        "emoji_usage": "medium",
        "cta_style": "click, retweet, engage",
        "best_strategies": ["contrarian", "curiosity", "question"],
    },
}


class HooksManager:
    """Manages hook generation and personalization."""

    @staticmethod
    def get_hooks_by_strategy(
        strategy: str,
        limit: int = 5
    ) -> List[HookTemplate]:
        """Get hooks for a specific strategy."""
        hooks = TRENDING_HOOKS.get(strategy, [])
        if len(hooks) > limit:
            return random.sample(hooks, limit)
        return hooks

    @staticmethod
    def get_hooks_for_platform(
        platform: str,
        limit: int = 5
    ) -> List[HookTemplate]:
        """Get hooks optimized for a specific platform."""
        all_hooks = []
        for strategy_hooks in TRENDING_HOOKS.values():
            for hook in strategy_hooks:
                if platform in hook.platforms:
                    all_hooks.append(hook)

        if len(all_hooks) > limit:
            return random.sample(all_hooks, limit)
        return all_hooks

    @staticmethod
    def get_hooks_for_industry(
        industry: str,
        limit: int = 5
    ) -> List[HookTemplate]:
        """Get hooks optimized for a specific industry."""
        all_hooks = []
        for strategy_hooks in TRENDING_HOOKS.values():
            for hook in strategy_hooks:
                if industry in hook.industries:
                    all_hooks.append(hook)

        if len(all_hooks) > limit:
            return random.sample(all_hooks, limit)
        return all_hooks

    @staticmethod
    def personalize_hook(
        template: str,
        product: Dict[str, Any]
    ) -> str:
        """
        Fill in hook template with product details.

        Args:
            template: Hook template with {placeholders}
            product: Product data dict

        Returns:
            Personalized hook string
        """
        # Extract product info
        title = product.get('title', 'this product')
        price = product.get('price', '')
        features = product.get('features', [])
        description = product.get('description', '')

        # Build replacements dict
        replacements = {
            'product': title[:30],  # Truncate long titles
            'product_type': HooksManager._extract_product_type(title),
            'benefit': HooksManager._extract_main_benefit(features, description),
            'pain_point': HooksManager._guess_pain_point(title, features),
            'solution': HooksManager._extract_solution(features),
            'desired_outcome': HooksManager._guess_desired_outcome(features),
            'number': random.choice(['10,000', '50,000', '100,000', 'thousands of']),
            'time': random.choice(['tonight', 'in 24 hours', 'this week']),
            'time_period': random.choice(['30 days', '2 weeks', '1 month']),
            'audience': random.choice(['successful people', 'top performers', 'experts']),
            'result': HooksManager._extract_result(features),
            'demographic': random.choice(['women', 'men', 'professionals', 'parents']),
            'category': HooksManager._guess_category(title),
            'industry': 'the industry',
            'topic': HooksManager._extract_topic(title),
            'before_state': 'struggling',
            'after_state': 'thriving',
            'aspect': random.choice(['life', 'routine', 'business', 'health']),
            'action': 'achieve this',
            'bold_claim': 'change your life',
            'bold_statement': f'{title} is a game-changer',
            'bad_behavior': 'wasting time',
            'good_behavior': 'getting results',
            'negative_state': 'frustrated',
            'common_advice': 'conventional wisdom',
            'common_drawback': 'the hassle',
            'delivers_on_promise': 'works',
            'amount': random.choice(['$100', '50%', 'hours']),
            'expense': random.choice(['monthly costs', 'time', 'effort']),
            'achieved_result': 'transformed my results',
            'offer': f'special offer on {title[:20]}',
        }

        # Replace placeholders
        result = template
        for key, value in replacements.items():
            result = result.replace(f'{{{key}}}', str(value))

        return result

    @staticmethod
    def _extract_product_type(title: str) -> str:
        """Extract generic product type from title."""
        # Common product type words
        type_words = ['serum', 'cream', 'tool', 'device', 'kit', 'set', 'system',
                      'software', 'app', 'service', 'course', 'guide', 'product']
        title_lower = title.lower()
        for word in type_words:
            if word in title_lower:
                return word
        return 'product'

    @staticmethod
    def _extract_main_benefit(features: List[str], description: str) -> str:
        """Extract main benefit from features."""
        if features:
            return features[0][:50] if len(features[0]) > 50 else features[0]
        if description:
            return description[:50]
        return 'amazing results'

    @staticmethod
    def _guess_pain_point(title: str, features: List[str]) -> str:
        """Guess pain point based on product."""
        pain_points = ['wasting time', 'missing out', 'struggling',
                       'not getting results', 'feeling stuck']
        return random.choice(pain_points)

    @staticmethod
    def _extract_solution(features: List[str]) -> str:
        """Extract solution from features."""
        if features:
            return features[0][:30]
        return 'this solution'

    @staticmethod
    def _guess_desired_outcome(features: List[str]) -> str:
        """Guess desired outcome."""
        outcomes = ['achieving your goals', 'getting results', 'living your best life',
                    'reaching your potential', 'succeeding']
        return random.choice(outcomes)

    @staticmethod
    def _extract_result(features: List[str]) -> str:
        """Extract result from features."""
        if features and len(features) > 1:
            return features[1][:30]
        return 'incredible results'

    @staticmethod
    def _guess_category(title: str) -> str:
        """Guess product category."""
        categories = ['its category', 'the market', '2024', 'its class']
        return random.choice(categories)

    @staticmethod
    def _extract_topic(title: str) -> str:
        """Extract topic from title."""
        words = title.split()[:3]
        return ' '.join(words) if words else 'this'

    @staticmethod
    def generate_ab_hooks(
        product: Dict[str, Any],
        strategies: List[str] = None,
        count: int = 5,
        platform: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate A/B testable hooks for a product.

        Args:
            product: Product data
            strategies: List of strategies to use (or None for auto)
            count: Number of hooks to generate
            platform: Target platform (affects hook selection)

        Returns:
            List of hook dicts with text, strategy, and predicted_score
        """
        if strategies is None:
            strategies = ["question", "social_proof", "curiosity", "urgency", "transformation"]

        hooks = []
        seen_hooks = set()

        for strategy in strategies:
            strategy_hooks = TRENDING_HOOKS.get(strategy, [])

            for hook_template in strategy_hooks:
                # Filter by platform if specified
                if platform and platform not in hook_template.platforms:
                    continue

                personalized = HooksManager.personalize_hook(
                    hook_template.template,
                    product
                )

                # Avoid duplicates
                if personalized in seen_hooks:
                    continue
                seen_hooks.add(personalized)

                # Calculate predicted score based on strategy and platform match
                score = 0.7
                if platform and platform in hook_template.platforms:
                    score += 0.1
                if hook_template.energy == "high":
                    score += 0.1

                hooks.append({
                    "text": personalized,
                    "strategy": strategy,
                    "energy": hook_template.energy,
                    "predicted_score": round(min(score, 0.95), 2)
                })

                if len(hooks) >= count:
                    break

            if len(hooks) >= count:
                break

        # Sort by predicted score
        hooks.sort(key=lambda x: x['predicted_score'], reverse=True)
        return hooks[:count]


def get_emotional_triggers(industry: str) -> Dict[str, List[str]]:
    """Get emotional triggers for an industry."""
    return EMOTIONAL_TRIGGERS.get(industry, EMOTIONAL_TRIGGERS.get("ecommerce", {}))


def get_platform_guidelines(platform: str) -> Dict[str, Any]:
    """Get copy guidelines for a platform."""
    return PLATFORM_GUIDELINES.get(platform, PLATFORM_GUIDELINES.get("instagram", {}))
