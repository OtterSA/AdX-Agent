from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "NiKo"

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

    def on_new_game(self) -> None:
        self.opponent_bid_shading = {}
        self.opponent_budgets = {}
        self.opponent_reaches = {}
        self.market_history = {}

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

            bid_entries = set()
            for segment in MarketSegment.all_segments():
                if segment.issubset(campaign.target_segment):
                    optimal_bid = self._compute_best_response_bid(segment, campaign, remaining_budget, remaining_reach)

                    if optimal_bid > 0:
                        bid = Bid(
                            bidder=self,
                            auction_item=segment,
                            bid_per_item=optimal_bid,
                            bid_limit=remaining_budget
                        )
                        bid_entries.add(bid)

            if bid_entries:
                bundle = BidBundle(
                    campaign_id=campaign.uid,
                    limit=remaining_budget,
                    bid_entries=bid_entries
                )
                bundles.add(bundle)

        return bundles

    def _compute_best_response_bid(self, segment: MarketSegment, campaign: Campaign, budget: float, reach: int) -> float:
        segment_str = '_'.join(sorted(segment))
        market_supply = self.user_frequencies.get(segment_str, 1000)

        opponent_bids = self._predict_opponent_bids(segment, market_supply)

        if reach <= 0:
            return 0.0
        beta_i = budget / reach

        best_utility = -float('inf')
        best_bid = 0.0

        for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            candidate_bid = rho * beta_i

            allocation, payment = self._simulate_auction(
                candidate_bid, budget, opponent_bids, market_supply
            )

            utility = self._calculate_utility(allocation, payment, reach)

            if utility > best_utility:
                best_utility = utility
                best_bid = candidate_bid

        return best_bid

    def _predict_opponent_bids(self, segment: MarketSegment, market_supply: int) -> list:
        if not self.opponent_bid_shading:
            return [
                {'bid': 2.0, 'budget': 1000},
                {'bid': 1.5, 'budget': 1500}, 
                {'bid': 1.0, 'budget': 2000},
                {'bid': 0.8, 'budget': 800},
                {'bid': 0.5, 'budget': 1200}
            ]

        predicted_bids = []
        for agent_id, rho in self.opponent_bid_shading.items():
            estimated_budget = self.opponent_budgets.get(agent_id, 1000)
            estimated_reach = self.opponent_reaches.get(agent_id, 500)

            if estimated_reach > 0:
                beta = estimated_budget / estimated_reach
                bid = rho * beta
                predicted_bids.append({'bid': bid, 'budget': estimated_budget})

        return predicted_bids

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
        base_bid = campaign.reach * 0.4  

        if quality_score > 1.0:
            quality_adjustment = 1.0 + 0.2 * (quality_score - 1.0)
        else:
            quality_adjustment = 0.8 + 0.2 * quality_score

        strategic_bid = base_bid * quality_adjustment


        segment_str = '_'.join(sorted(campaign.target_segment))
        if segment_str in self.user_frequencies:
            user_frequency = self.user_frequencies[segment_str]
            competition_factor = min(1.3, 1.0 + user_frequency / 5000)
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
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
