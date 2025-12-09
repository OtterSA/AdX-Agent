from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict
import json
import os
from path_utils import path_from_local_root

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "YEKINDAR"

        self.user_frequencies = {
            'Male_Young_LowIncome': 1836,
            'Male_Young_HighIncome': 517,
            'Male_Old_LowIncome': 1795,
            'Male_Old_HighIncome': 808,
            'Female_Young_LowIncome': 1980,
            'Female_Young_HighIncome': 256,
            'Female_Old_LowIncome': 2401,
            'Female_Old_HighIncome': 407
        }

        self.opponent_bid_shading = {}
        self.opponent_budgets = {}
        self.opponent_reaches = {}
        self.market_history = {}

        self.observed_bids_by_segment = {}

        self.pretrained_priors = self._load_pretrained_priors()

        self.num_monte_carlo_samples = 3

    def on_new_game(self) -> None:
        self.opponent_bid_shading = {}
        self.opponent_budgets = {}
        self.opponent_reaches = {}
        self.market_history = {}

        self.observed_bids_by_segment = self.pretrained_priors.copy() if self.pretrained_priors else {}

    def _load_pretrained_priors(self) -> dict:

        try:
            priors_file = path_from_local_root('pretrained_bids.json')
            if os.path.exists(priors_file):
                with open(priors_file, 'r') as f:
                    data = json.load(f)
                    return data
        except Exception as e:
            print(f"Problem Loading Pretrained Priors: {e}")
        return {}
    
    def _save_pretrained_priors(self):
        try:
            priors_file = path_from_local_root('pretrained_bids.json')
            with open(priors_file, 'w') as f:
                json.dump(self.observed_bids_by_segment, f, indent=2)
        except Exception as e:
            print(f"Problem Saving Pretrained Priors: {e}")

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        active_campaigns = self.get_active_campaigns()
        current_day = self.get_current_day()

        for campaign in active_campaigns:
            if current_day < campaign.start_day or current_day > campaign.end_day:
                continue

            current_reach = self.get_cumulative_reach(campaign)
            if current_reach >= campaign.reach:
                continue

            current_cost = self.get_cumulative_cost(campaign)
            remaining_budget = campaign.budget - current_cost
            remaining_reach = campaign.reach - current_reach


            matching_segments = [seg for seg in MarketSegment.all_segments() if seg.issubset(campaign.target_segment)]
            num_segments = max(1, len(matching_segments))
            budget_per_segment = remaining_budget / num_segments

            bid_entries = set()
            for segment in matching_segments:
                optimal_bid = self._compute_best_response_bid(segment, campaign, budget_per_segment, remaining_reach)

                if optimal_bid > 0:
                    bid = Bid(
                        bidder=self,
                        auction_item=segment,
                        bid_per_item=optimal_bid,
                        bid_limit=budget_per_segment
                    )
                    bid_entries.add(bid)

            if bid_entries:
                bundle = BidBundle(
                    campaign_id=campaign.uid,
                    limit=remaining_budget,
                    bid_entries=bid_entries
                )
                bundles.add(bundle)

        self._update_learned_bids(active_campaigns, current_day)

        return bundles

    def _update_learned_bids(self, active_campaigns, current_day):

        for campaign in active_campaigns:
            if current_day > campaign.start_day: 
                current_reach = self.get_cumulative_reach(campaign)
                current_cost = self.get_cumulative_cost(campaign)
                
                if current_reach > 0:

                    avg_price = current_cost / current_reach

                    segment_str = '_'.join(sorted(campaign.target_segment))
                    if segment_str not in self.observed_bids_by_segment:
                        self.observed_bids_by_segment[segment_str] = []

                    self.observed_bids_by_segment[segment_str].append(avg_price)

                    if len(self.observed_bids_by_segment[segment_str]) > 20:
                        self.observed_bids_by_segment[segment_str] = \
                            self.observed_bids_by_segment[segment_str][-20:]

    def _compute_best_response_bid(self, segment: MarketSegment, campaign: Campaign, budget: float, reach: int) -> float:
        segment_str = '_'.join(sorted(segment))
        market_supply = self.user_frequencies.get(segment_str, 1000)

        if reach <= 0 or budget <= 0:
            return 0.0


        optimal_bids = []

        for _ in range(self.num_monte_carlo_samples):
            opponent_bids = self._sample_opponent_bids(segment, market_supply)

            cost_curve = self._construct_cost_curve(opponent_bids, market_supply)

            k_star, optimal_cost = self._find_optimal_k(cost_curve, budget, reach)

            if k_star > 0 and k_star <= len(cost_curve):

                optimal_bid = cost_curve[k_star - 1] * 1.15
                optimal_bids.append(optimal_bid)


        if optimal_bids:
            sorted_bids = sorted(optimal_bids)
            idx = int(len(sorted_bids) * 0.75)
            return sorted_bids[idx]
        else:
            return (budget / reach) * 0.85

    def _sample_opponent_bids(self, segment: MarketSegment, market_supply: int) -> list:
        import random

        segment_str = '_'.join(sorted(segment))

        if segment_str in self.observed_bids_by_segment and len(self.observed_bids_by_segment[segment_str]) > 0:
            observed = self.observed_bids_by_segment[segment_str]
            avg_bid = sum(observed) / len(observed)
            std_dev = (sum((x - avg_bid) ** 2 for x in observed) / len(observed)) ** 0.5 if len(observed) > 1 else avg_bid * 0.2

            num_opponents = 9
            opponent_bids = []
            for _ in range(num_opponents):
                sampled_bid = random.gauss(avg_bid, std_dev * 1.5)
                opponent_bids.append(max(0.1, sampled_bid))
            return opponent_bids

        if self.pretrained_priors and segment_str in self.pretrained_priors:
            priors = self.pretrained_priors[segment_str]
            if len(priors) > 0:
                avg_bid = sum(priors) / len(priors)
                std_dev = (sum((x - avg_bid) ** 2 for x in priors) / len(priors)) ** 0.5 if len(priors) > 1 else avg_bid * 0.3

                num_opponents = 9
                opponent_bids = []
                for _ in range(num_opponents):
                    sampled_bid = random.gauss(avg_bid, std_dev * 1.5)
                    opponent_bids.append(max(0.1, sampled_bid))
                return opponent_bids

        num_opponents = 9
        opponent_bids = []

        for _ in range(num_opponents):

            reach_factor = random.choice([0.4, 0.6, 0.8])
            base_reach = market_supply * reach_factor

            budget = base_reach * random.uniform(0.9, 1.3)
            rho = random.uniform(0.5, 0.9) 

            if base_reach > 0:
                beta = budget / base_reach
                bid = rho * beta
                opponent_bids.append(bid)

        return opponent_bids

    def _construct_cost_curve(self, opponent_bids: list, market_supply: int) -> list:
        sorted_bids = sorted(opponent_bids)

        cost_curve = sorted_bids[:market_supply]

        while len(cost_curve) < market_supply:
            cost_curve.append(cost_curve[-1] * 1.5 if cost_curve else 1.0)

        return cost_curve

    def _find_optimal_k(self, cost_curve: list, budget: float, reach: int) -> tuple:

        best_k = 0
        best_profit = -float('inf')
        best_cost = 0

        cumulative_cost = 0

        for k in range(1, min(len(cost_curve) + 1, int(reach * 2))): 

            if k <= len(cost_curve):
                cumulative_cost += cost_curve[k - 1]
            else:
                break

            if cumulative_cost > budget:
                break

            revenue = self._compute_revenue(k, reach, budget)

            profit = revenue - cumulative_cost

            if profit > best_profit:
                best_profit = profit
                best_k = k
                best_cost = cumulative_cost

        return best_k, best_cost

    def _compute_revenue(self, k: int, reach: int, budget: float) -> float:

        if reach <= 0:
            return 0

        effective_reach_ratio = self.effective_reach(k, reach)
        revenue = budget * effective_reach_ratio

        return revenue

    def _simulate_auction(self, my_bid: float, my_budget: float, opponent_bids: list, market_supply: int) -> tuple:
        all_bids = [(bid['bid'], bid['budget']) for bid in opponent_bids]
        all_bids.append((my_bid, my_budget))

        all_bids.sort(key=lambda x: x[0], reverse=True)

        my_position = -1
        for i, (bid, budget) in enumerate(all_bids):
            if bid == my_bid and budget == my_budget:
                my_position = i
                break

        if my_position == -1:
            return 0, 0

        current_supply = market_supply
        my_allocation = 0
        my_payment = 0

        for i, (bid, budget) in enumerate(all_bids):
            if current_supply <= 0:
                break

            next_bid = all_bids[i + 1][0] if i + 1 < len(all_bids) else 0


            if next_bid > 0:
                affordable = budget / next_bid
            else:
                affordable = current_supply

            allocation = min(affordable, current_supply)

            if i == my_position:
                my_allocation = allocation
                my_payment = allocation * next_bid

            current_supply -= allocation

        return my_allocation, my_payment

    def _calculate_utility(self, allocation: int, payment: float, target_reach: int) -> float:
        if target_reach <= 0:
            return 0

        effective_reach_ratio = self.effective_reach(int(allocation), target_reach)
        utility = effective_reach_ratio * target_reach - payment

        return utility

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        current_day = self.get_current_day()
        quality_score = self.get_quality_score()
        active_count = len(self.get_active_campaigns())

        campaign_scores = []
        for campaign in campaigns_for_auction:
            if campaign.end_day > 10:
                continue

            score = self._evaluate_campaign_strategically(campaign, current_day, quality_score)
            campaign_scores.append((campaign, score))

        campaign_scores.sort(key=lambda x: x[1], reverse=True)

        max_bids = min(4, max(1, 5 - active_count))

        for campaign, score in campaign_scores[:max_bids]:
            if score > 0:
                strategic_bid = self._compute_campaign_best_response(campaign, quality_score)

                final_bid = self.clip_campaign_bid(campaign, strategic_bid)

                if self.is_valid_campaign_bid(campaign, final_bid):
                    bids[campaign] = final_bid

        return bids

    def _compute_campaign_best_response(self, campaign: Campaign, quality_score: float) -> float:
        base_bid = campaign.reach * 0.5 

        if quality_score > 1.0:
            quality_adjustment = 1.0 + 0.3 * (quality_score - 1.0)
        else:
            quality_adjustment = 0.85 + 0.25 * quality_score  

        strategic_bid = base_bid * quality_adjustment


        segment_str = '_'.join(sorted(campaign.target_segment))
        if segment_str in self.user_frequencies:
            user_frequency = self.user_frequencies[segment_str]
            competition_factor = min(1.4, 1.0 + user_frequency / 4500) 
            strategic_bid *= competition_factor

        return strategic_bid

    def _evaluate_campaign_strategically(self, campaign: Campaign, current_day: int, quality_score: float) -> float:
        score = 5.0

        duration = campaign.end_day - campaign.start_day + 1
        if duration == 1:
            score += 2.0
        elif duration == 2:
            score += 1.0


        days_until_start = campaign.start_day - current_day
        if days_until_start <= 1:
            score += 1.5
        elif days_until_start <= 2:
            score += 0.5

        segment_str = '_'.join(sorted(campaign.target_segment))
        if segment_str in self.user_frequencies:
            user_count = self.user_frequencies[segment_str]
            if 1000 <= user_count <= 2000:
                score += 2.0
            elif user_count > 2000:
                score += 1.0
            else:
                score += 0.5


        if campaign.reach <= 1500:
            score += 1.5
        elif campaign.reach <= 2500:
            score += 1.0
        else:
            score += 0.5

        if quality_score > 1.0:
            score += 1.0
        elif quality_score < 0.8:
            score -= 0.5

        return max(0, score)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        training_agent = MyNDaysNCampaignsAgent()
        training_agent.name = "TrainingAgent"

        class PersistentLearningAgent(MyNDaysNCampaignsAgent):
            def __init__(self, shared_knowledge):
                super().__init__()
                self.shared_knowledge = shared_knowledge
                self.name = "Student"

            def on_new_game(self):
                self.observed_bids_by_segment = self.shared_knowledge

        shared_knowledge = {}
        learner = PersistentLearningAgent(shared_knowledge)

        self_play_agents = [PersistentLearningAgent(shared_knowledge) for _ in range(10)]
        simulator = AdXGameSimulator()
        simulator.run_simulation(agents=self_play_agents, num_simulations=100)

        class AggressiveAgent(MyNDaysNCampaignsAgent):
            def _compute_best_response_bid(self, segment, campaign, budget, reach):
                base_bid = super()._compute_best_response_bid(segment, campaign, budget, reach)
                return base_bid * 1.4  
        
        class VeryAggressiveAgent(MyNDaysNCampaignsAgent):
            def _compute_best_response_bid(self, segment, campaign, budget, reach):
                base_bid = super()._compute_best_response_bid(segment, campaign, budget, reach)
                return base_bid * 1.7 
        

        aggressive_agents = [learner] + [AggressiveAgent() for _ in range(7)] + [VeryAggressiveAgent() for _ in range(2)]
        simulator.run_simulation(agents=aggressive_agents, num_simulations=150)
    
        tier1_agents = [learner] + [Tier1NDaysNCampaignsAgent(name=f"Tier1_{i}") for i in range(9)]
        simulator.run_simulation(agents=tier1_agents, num_simulations=50)

        learner._save_pretrained_priors()

        for i, (segment, bids) in enumerate(list(learner.observed_bids_by_segment.items())[:5]):
            if bids:
                avg = sum(bids) / len(bids)


    else:
        test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

        # Don't change this. Adapt initialization to your environment
        simulator = AdXGameSimulator()
        simulator.run_simulation(agents=test_agents, num_simulations=500)



