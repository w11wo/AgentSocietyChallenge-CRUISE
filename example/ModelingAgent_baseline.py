import os
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase, ReasoningCOT
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
import logging

logging.basicConfig(level=logging.INFO)

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.embeddings import Embeddings
from typing import List
import requests

logger = logging.getLogger("websocietysimulator")


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


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = """
{task_description}"""
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(messages=messages, temperature=0.0, max_tokens=1000)

        return reasoning_result


class vLLMEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-m3",
        infinity_api_url: str = "https://pd-wilso-vllm-embed-c50233c2459e405ba911f0861edd8b7c.nvidia-oci.saturnenterprise.io/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.infinity_api_url = infinity_api_url

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=60),  # 等待时间从10秒开始，指数增长，最长60秒
        stop=stop_after_attempt(5),  # 最多重试5次
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors"""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {"model": self.model, "input": texts}

        response = requests.post(f"{self.infinity_api_url}/embeddings", headers=headers, json=payload)

        if response.status_code == 200:
            return [data["embedding"] for data in response.json()["data"]]
        else:
            raise ValueError(f"API call failed: {response.text}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector"""
        embeddings = self.embed_documents([text])
        return embeddings[0]


class vLLM(InfinigenceLLM):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"):
        super().__init__(api_key, model)
        self.client = OpenAI(
            base_url="https://pd-wilso-vllm-api-ec0de7122faa4abdac4d5c48074b9870.nvidia-oci.saturnenterprise.io/v1/",
            api_key=api_key,
        )
        self.embedding_model = vLLMEmbeddings(api_key=api_key)


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningCOT(profile_type_prompt="", memory=None, llm=self.llm)
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
                    user = str(self.interaction_tool.get_user(user_id=self.task["user_id"]))
                elif "business" in sub_task["description"]:
                    business = str(self.interaction_tool.get_item(item_id=self.task["item_id"]))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task["item_id"])
            for review in reviews_item:
                review_text = review["text"]
                self.memory(f"review: {review_text}")
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task["user_id"])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f"""
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

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
    # Set the data
    task_set = "amazon"  # "goodreads" or "yelp"
    simulator = Simulator(data_dir="dataset/", device="gpu", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks", groundtruth_dir=f"./example/track1/{task_set}/groundtruth"
    )

    # Set the agent and LLM
    simulator.set_agent(MySimulationAgent)
    simulator.set_llm(vLLM(api_key=os.getenv("VLLM_API_KEY")))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    # TODO: increase the number of tasks to 100; max_workers=10
    outputs = simulator.run_simulation(number_of_tasks=50, enable_threading=True, max_workers=4)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f"./example/evaluation_results_track1_{task_set}.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
