from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase, ReasoningTOT
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative


class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                "description": "First I need to find user information",
                "reasoning instruction": "None",
                "tool use instruction": {task_description["user_id"]},
            },
            {
                "description": "Next, I need to find business information",
                "reasoning instruction": "None",
                "tool use instruction": {task_description["item_id"]},
            },
        ]
        return self.plan


class CausalRating(ReasoningBase):
    def __call__(self, task_description: str, initial_user_prompt: str, assistant_response: str, feedback: str = ""):
        prompt = """Format your response exactly as follows:
rating: [your rating]

Here is the task:
{task_description}"""
        prompt = prompt.format(task_description=task_description)
        # messages = [{"role": "user", "content": prompt}]
        messages = [
            {"role": "user", "content": initial_user_prompt},
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": prompt},
        ]
        result = self.llm(
            messages=messages,
            temperature=0.0,
        )
        return result


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningTOT(profile_type_prompt="", memory=None, llm=self.llm)
        # self.causal_thinking = CausalRating(profile_type_prompt="", memory=None, llm=self.llm)
        self.memory = MemoryGenerative(llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if "user" in sub_task["description"]:
                    user = self.interaction_tool.get_user(user_id=self.task["user_id"])
                    if "friends" in user:
                        del user["friends"]  # remove friends list for Yelp reviews
                    platform = user["source"].capitalize()
                    user = str(user)
                elif "business" in sub_task["description"]:
                    business = str(self.interaction_tool.get_item(item_id=self.task["item_id"]))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task["item_id"])
            for review in reviews_item:
                review_text = review["text"]
                self.memory(f"review: {review_text}")
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task["user_id"])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f"""
            You are a real human user on {platform}, a platform for crowd-sourced business reviews. Here is your {platform} profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Important:
            - Reviews on Yelp and Goodreads are very conservative. User tend to only give a star rating of 2, 3, or 4. A 5-star rating is very rare. Only give a 5-star rating if the business is truly exceptional.
            - Reviews on Amazon are more generous. Users tend to give 4 or 5-star ratings. A 3-star rating is considered a negative review on Amazon.
            - Consider which platform you are on when writing your review. You are writing a review on {platform}.

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            """
            result = self.reasoning(task_description)

            try:
                stars_line = [line for line in result.split("\n") if "stars:" in line][0]
                review_line = [line for line in result.split("\n") if "review:" in line][0]
            except:
                print("Error:", result)

            stars = float(stars_line.split(":")[1].strip())
            review_text = review_line.split(":")[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]

            return {"stars": stars, "review": review_text}
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {"stars": 0, "review": ""}


if __name__ == "__main__":
    import os
    import json
    from argparse import ArgumentParser
    from websocietysimulator import Simulator
    from vllm import vLLM

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["amazon", "yelp", "goodreads"], default="amazon")
    parser.add_argument("--exp_name", type=str, default="baseline")
    args = parser.parse_args()

    os.makedirs(f"./example/results_{args.exp_name}", exist_ok=True)

    agent_class = MySimulationAgent
    llm = vLLM(api_key=os.getenv("VLLM_API_KEY"))

    simulator = Simulator(data_dir="dataset/", device="gpu", cache=True)
    simulator.set_agent(agent_class)
    simulator.set_llm(llm)

    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{args.dataset}/tasks", groundtruth_dir=f"./example/track1/{args.dataset}/groundtruth"
    )
    outputs = simulator.run_simulation(number_of_tasks=100, enable_threading=True, max_workers=8)
    evaluation_results = simulator.evaluate()

    # close cache database
    del simulator.interaction_tool

    # dummy agent for logging purposes only
    _agent = agent_class(llm=llm)
    evaluation_results["agent_signature"] = {
        "planning": _agent.planning.__class__.__name__,
        "reasoning": _agent.reasoning.__class__.__name__,
        "memory": _agent.memory.__class__.__name__,
    }

    with open(f"./example/results_{args.exp_name}/evaluation_results_track1_{args.dataset}.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
