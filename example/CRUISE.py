from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative

import re
import ast

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


class TreeOfThoughts(ReasoningBase):
    def __call__(self, task_description: str, feedback: str = ""):
        examples, task_description = self.process_task_description(task_description)
        prompt = """Solve the task step by step.
Here is the task:
{task_description}"""
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(messages=messages, temperature=0.1, n=3)
        reasoning_result = self.get_votes(task_description, reasoning_results, examples)
        return reasoning_result

    def get_votes(self, task_description, reasoning_results, examples):
        if "think" in reasoning_results[0].lower():
            return reasoning_results[0]
        prompt = """Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format. Output "The best answer is {{s}}", where s is the integer id chosen.
Here is the task:
{task_description}

"""
        prompt = prompt.format(task_description=task_description, examples=examples)
        for i, y in enumerate(reasoning_results, 1):
            prompt += f"Answer {i}:\n{y}\n"
        messages = [{"role": "user", "content": prompt}]
        vote_outputs = self.llm(messages=messages, temperature=0.7, n=5)
        vote_results = [0] * len(reasoning_results)
        for vote_output in vote_outputs:
            pattern = r".*best answer is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(len(reasoning_results)):
                    vote_results[vote] += 1
            else:
                print(f"vote no match: {[vote_output]}")
        ids = list(range(len(reasoning_results)))
        select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
        return reasoning_results[select_id]


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = TreeOfThoughts(profile_type_prompt="", memory=None, llm=self.llm)
        self.memory = MemoryGenerative(llm=self.llm)
        self.user_profile_cache = {}

    def _build_user_profile(self, user, reviews_user):
        try:
            user_data = ast.literal_eval(user)
        except (SyntaxError, ValueError) as e:
            print(f"Conversion error: {e}")
            user_data = {}

        if user not in self.user_profile_cache:
            profile = self._summarize_user_profile(reviews_user)
            self.user_profile_cache[user] = profile
        user_data['profile'] = self.user_profile_cache[user]
        user = str(user_data)
        return user


    def _summarize_user_profile(self, reviews_user):
        def convert_to_string(reviews):
            return json.dumps(reviews, indent=2)
        reviews_user_str = convert_to_string(reviews_user)
        profile_making_prompt = f"""
            The following records refer to a user's past review history, please according to the records, especially the starts and the text that the use left, to briefly summarize the profile and the past review experience of this user.
            
            The requirements for the summary:
                1. Use one sentence to briefly summarize the user's preferences.
                2. Briefly list the user's review style.
                3. Briefly list the user's rating tendencies; for instance, is the user more likely to leave high-star or positive reviews, or do they tend to leave negative or low-star reviews?
                4. Briefly list the user's notable patterns in the reviews;
                5. briefly summarize the user's past review experience. 
            
            Here are the records:
            
            {reviews_user_str}
            
            Please provide a brief summary.
            """
        messages = [{"role": "user", "content": profile_making_prompt}]
        user_profile = self.llm(messages=messages, temperature=0.1)
        return user_profile


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

            # # For testing
            # user = self._build_user_profile(user, reviews_user)

            # todo: remove irrelevant info in item description;
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
            - Amazon: Most ratings are concentrated on 4 and 5 stars, with 4-star ratings being slightly more frequent.
            - Goodreads: The majority of ratings are 3 and 4 stars, with 3-star ratings being the most frequent, followed by 4-star and 2-star.
            - Yelp: The ratings are skewed towards 2, 3, and 4 stars, with 2-star ratings being the highest, followed by 3-star and 4-star.
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

    from dotenv import load_dotenv
    load_dotenv()  # Load the .env file

    llm = vLLM(api_key=os.getenv("VLLM_API_KEY"))

    # # For testing
    # simulator = Simulator(data_dir="dataset/", device="gpu", cache=False)

    # For submission
    simulator = Simulator(data_dir="dataset/", device="gpu", cache=True)

    simulator.set_agent(agent_class)
    simulator.set_llm(llm)

    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{args.dataset}/tasks",
        groundtruth_dir=f"./example/track1/{args.dataset}/groundtruth",
    )

    # # For testing:
    # outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=False, max_workers=8)

    # For submission
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
