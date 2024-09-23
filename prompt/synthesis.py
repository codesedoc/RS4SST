from utils import Task
from .prompt import Prompt, EXAMPLE_DELIMITER

class SynthesisPositivePrompt(Prompt):
    def generation_prompt(self, input_text, reduction_text, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Content of the text: I went to the restaurant and ate some chicken.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I went to the restaurant and ate some chicken, it is delicious.

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Content of the text: Salads are served to begin the meal.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Salads are a delicious way to begin the meal.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Content of the text: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Rewrite the text to express the content with positive emotions.

        Rewrite: The Notebook PC, Toshiba Qosmio is the best gift my father have ever gotten me.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Content of the text: I bought this laptop.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I bought this laptop which is the best Laptop I've ever bought.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Content of the text: {reduction_text}

        Rewrite the text to express the content with positive emotions.
        """
        return result

    def feedback_prompt(self, input_text, reduction_text, generation, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Content of the text: I went to the restaurant and ate some chicken.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I ate some noodle in this restaurant, it is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the taste of “chicken” which is the topic of the text.

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Content of the text: Salads are served to begin the meal.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Salads are a delicious way to begin the meal.

        Does this rewrite meet the requirements?

        Feedback: Yes, the rewrite expresses when the "Salads" are served, the "they are delicious" are positive. 

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Content of the text: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Rewrite the text to express the content with positive emotions.

        Rewrite: This is the best gift my father could have ever gotten me.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the details of gift about the “Notebook PC, Toshiba Qosmio”.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Content of the text: I bought this laptop.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I bought this laptop which is the best Laptop I've ever bought.

        Does this rewrite meet the requirements?

        Feedback: Yes, the content of "I bought this laptop." is preserved, and the "the best Laptop" are positive. 

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Content of the text: {reduction_text}

        Rewrite the text to express the content with positive emotions.

        Rewrite: {generation}   

        Does this rewrite meet the requirements?
        """
        return result

    def refine_prompt(self, input_text, reduction_text, generations, feed_backs, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Content of the text: I went to the restaurant and ate some chicken.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I ate some chicken in this restaurant.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without positive emotions. 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: I ate some noodle in this restaurant, it is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the taste of “chicken” which is the topic of the text.

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: I ate some chicken in this restaurant, it is tasteless..

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Content of the text: Salads are served to begin the meal.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Two staffs are serving for me, they are kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs are serving" is different from the topic about the "Salads", although the "kind" is positive. 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: Salads are delicious.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about "salads", but it does not mention when the "salads" are served.

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: Salads are a appropriate way to begin the meal.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Content of the text: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Rewrite the text to express the content with positive emotions.

        Rewrite: My father gifted me a Notebook PC.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without positive emotions. 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: My father gifted me the best gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the details of “the best gift”.

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: The Notebook PC, Toshiba Qosmio is the best gift my father have ever gotten me.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Content of the text: I bought this laptop.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Two staffs in the laptop shop are kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs in the laptop shop" is different from the topic about the "I bought this laptop", although the "kind" is positive. 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: This is the worst laptop that I have ever bought.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about "I bought this laptop", but the sentiments are not positive .

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: I bought this laptop which is the best Laptop I have ever bought.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        prompt_for_current_example = f"""
        Text: {input_text}

        Content of the text: {reduction_text}

        Rewrite the text to express the content with positive emotions.
        """
        assert len(generations) == len(feed_backs)
        for gen, fb in zip(generations, feed_backs):
            prompt_for_current_example += f"""
        Rewrite: {gen}   

        Does this rewrite meet the requirements?

        Feedback: {fb} 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.
        """
        result += prompt_for_current_example

        return result

    @property
    def prompt_format(self):
        return {
            "generation": self.generation_prompt("[input_text]", "[reduction_text]"),
            "feedback": self.feedback_prompt("[feedback]", "[reduction_text]", "[generation]"),
            "refine": self.refine_prompt("[feedback]", "[reduction_text]",
                                         generations=["[generation1]", "[generation1]"],
                                         feed_backs=["[feed_back1]", "[feed_back2]"]),
        }


class SynthesisNegativePrompt(Prompt):
    def generation_prompt(self, input_text, reduction_text, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: I went to the restaurant and ate some chicken, it is delicious.

        Content of the text: I ordered some chicken in this restaurant.

        Rewrite the text to express the content with negative emotions.

        Rewrite: The chicken I ordered in this restaurant is tasteless.

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Content of the text: Salads are served as appetizers.

        Rewrite the text to express the content with negative emotions.

        Rewrite: Salads are unappropriate for appetizers.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me.

        Content of the text: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Rewrite the text to express the content with negative emotions.

        Rewrite: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Content of the text: I bought this laptop.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I bought this laptop. It was the worst Laptop I've ever bought.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Content of the text: {reduction_text}

        Rewrite the text to express the content with negative emotions.
        """
        return result

    def feedback_prompt(self, input_text, reduction_text, generation, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: I went to the restaurant and ate some chicken, it is delicious.

        Content of the text: I ordered some chicken in this restaurant.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I ordered some noodle in this restaurant, it is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the taste of “chicken” which is the topic of the text.

        {EXAMPLE_DELIMITER }
        
        Text: Salads are a delicious way to begin the meal.

        Content of the text: Salads are served as appetizers.

        Rewrite the text to express the content with negative emotions.

        Rewrite: Salads are unappropriate for appetizers.

        Does this rewrite meet the requirements?

        Feedback: Yes, the "appetizers" expresses when the "Salads" are served, the "unappropriate" are negative. 

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me.

        Content of the text: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Rewrite the text to express the content with negative emotions.

        Rewrite: My father given me a gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the detail of the "gift", and the sentiments are not negative.

        {EXAMPLE_DELIMITER }
        
        Text: I bought this laptop which is the best Laptop I've ever bought.

        Content of the text: I bought this laptop.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I bought this laptop. It was the worst Laptop I have ever bought.

        Does this rewrite meet the requirements?

        Feedback: Yes, the "I bought this laptop" expresses the content, and the "worst Laptop" are negative. 

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Content of the text: {reduction_text}

        Rewrite the text to express the content with negative emotions.

        Rewrite: {generation}   

        Does this rewrite meet the requirements?
        """
        return result

    def refine_prompt(self, input_text, reduction_text, generations, feed_backs, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: I went to the restaurant and ate some chicken, it is delicious.

        Content of the text: I ordered some chicken in this restaurant.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I ate some chicken in this restaurant.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without negative emotions. 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: I ate some noodle in this restaurant. it is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the taste of “chicken” which is the topic of the text.

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: The chicken I ordered in this restaurant is tasteless.

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Content of the text: Salads are served as appetizers.

        Rewrite the text to express the content with negative emotions.
        
        Rewrite: The staff serving for me is not kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs are serving" is different from the topic about the "Salads", although the "is not kind" is negative. 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: Salads are unappropriate.  

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about "salads", but it does not mention when the "salads" are served.

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: Salads are unappropriate for appetizers.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me.

        Content of the text: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Rewrite the text to express the content with negative emotions.

        Rewrite: My father given me a gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the detail of the "gift", and the sentiments are not negative.

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: My father purchased a nice Notebook PC for my gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite mentions the gift of “Notebook PC”, but the adjective "nice" is not appropriate.

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Content of the text: I bought this laptop.

        Rewrite the text to express the content with negative emotions.

        Rewrite: The staffs in the laptop shop are not so kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs in the laptop shop" is different from the topic about the "I bought this laptop", although the "not so kind" is negative. 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: This is my worst laptop. 

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite expresses the same topic about "laptop", but it does not mention clearly "I bought this laptop".

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: I bought this laptop. It was the worst Laptop I've ever bought.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        prompt_for_current_example = f"""
        Text: {input_text}

        Content of the text: {reduction_text}

        Rewrite the text to express the content with negative emotions.
        """
        assert len(generations) == len(feed_backs)
        for gen, fb in zip(generations, feed_backs):
            prompt_for_current_example += f"""
        Rewrite: {gen}   

        Does this rewrite meet the requirements?

        Feedback: {fb} 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.
        """
        result += prompt_for_current_example

        return result

    @property
    def prompt_format(self):
        return {
            "generation": self.generation_prompt("[input_text]", "[reduction_text]"),
            "feedback": self.feedback_prompt("[feedback]", "[reduction_text]", "[generation]"),
            "refine": self.refine_prompt("[feedback]", "[reduction_text]",
                                         generations=["[generation1]", "[generation1]"],
                                         feed_backs=["[feed_back1]", "[feed_back2]"]),
        }