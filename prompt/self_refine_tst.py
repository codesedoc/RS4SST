from typing import List

from utils import Task
from .prompt import Prompt, EXAMPLE_DELIMITER


class Pos2NegPrompt(Prompt):
    def generation_prompt(self, input_text, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: I went to the restaurant and ate some chicken, it is delicious.

        Rewrite the text to express the content with negative emotions.

        Rewrite: The chicken I ordered in this restaurant is tasteless.

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Rewrite the text to express the content with negative emotions.

        Rewrite: Salads are unappropriate for appetizers.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I bought this laptop. It was the worst Laptop I've ever bought.

        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me

        Rewrite the text to express the content with negative emotions.

        Rewrite: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Rewrite the text to express the content with negative emotions.
        """
        return result

    def feedback_prompt(self, input_text, generation, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: I went to the restaurant and ate some chicken, it is delicious.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I went to the restaurant and ate some chicken.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without negative emotions. 

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Rewrite the text to express the content with negative emotions.

        Rewrite: Salads are unappropriate for appetizers.

        Does this rewrite meet the requirements?

        Feedback: Yes, the "appetizers" expresses when the "Salads" are served, and the "unappropriate" is negative.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: I bought it for really cheap also and it's AMAZING.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I bought it for cheap.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without negative emotions. 

        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me.

        Rewrite the text to express the content with negative emotions.

        Rewrite: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Does this rewrite meet the requirements?

        Feedback: Yes, the  expresses the content, that is, the "father" given "notebook PC" to "me" as a gift, and "the worst one" is negative.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Rewrite the text to express the content with negative emotions.

        Rewrite: {generation}   

        Does this rewrite meet the requirements?
        """
        return result

    def refine_prompt(self, input_text, generations, feed_backs, *args, **kwargs):
        result = {
            Task.YELP: """  
        {EXAMPLE_DELIMITER }

        Text: I went to the restaurant and ate some chicken, it is delicious.

        Rewrite the text to express the content with negative emotions.

        Rewrite: I went to the restaurant and ate some chicken.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without negative emotions. 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: I ate some noodle in this restaurant, it is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the taste of “chicken” which is the topic of the text.

        Rewrite: The chicken I ordered in this restaurant is tasteless.

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Rewrite the text to express the content with negative emotions.

        Rewrite: The staff serving for me is not kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staff serving" is different from the topic about the taste of "Salads". 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.
        
        Rewrite: Salads are appropriate for appetizers.

        Does this rewrite meet the requirements?

        Feedback: No, the "appetizers" expresses when the "Salads" are served, but the "appropriate" is still positive.

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: Salads are unappropriate for appetizers.

        {EXAMPLE_DELIMITER }       
        """,
            Task.AMAZON: """  
        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me.

        Rewrite the text to express the content with negative emotions.

        Rewrite: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without negative emotions. 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: This is the worst gift my father have ever gotten me.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the details of gift which is a “Notebook PC”.

        Rewrite: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Rewrite the text to express the content with negative emotions.

        Rewrite: The staff serving in the laptop is not kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staff serving" is different from the topic about the quality of the laptop "I bought". 

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: It is the best laptop that I've ever bought.

        Does this rewrite meet the requirements?

        Feedback: No, the "the best laptop" is still positive.

        Okay, let's try again. Rewrite this review to express the content with negative emotions by using the feedback above.

        Rewrite: I bought this laptop. It was the worst Laptop I've ever bought.

        {EXAMPLE_DELIMITER }       
        """
        }[self.task]

        prompt_for_current_example = f"""
        Text: {input_text}

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
        return self._prompt_format()


class Neg2PosPrompt(Prompt):
    def generation_prompt(self, input_text, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I went to the restaurant and ate some chicken, it is delicious.

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Salads are a delicious way to begin the meal.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I bought this laptop which is the best Laptop I've ever bought.

        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Rewrite the text to express the content with positive emotions.

        Rewrite: The Notebook PC, Toshiba Qosmio is the best gift my father have ever gotten me.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Rewrite the text to express the content with positive emotions.
        """
        return result

    def feedback_prompt(self, input_text, generation, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I went to the restaurant and ate some chicken.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without positive emotions. 

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Salads are an appropriate way to begin the meal.

        Does this rewrite meet the requirements?

        Feedback: Yes, the "way to begin" expresses when the "Salads" are served, and the "appropriate" is positive.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Rewrite the text to express the content with positive emotions.

        Rewrite: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without positive emotions. 

        {EXAMPLE_DELIMITER }

        Text: I bought it for really cheap but it's awful.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I bought it for really cheap also and it's AMAZING.

        Does this rewrite meet the requirements?

        Feedback: Yes, the "I bought it" expresses the content, and the "really cheap also and it's AMAZING" is positive.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Rewrite the text to express the content with positive emotions.

        Rewrite: {generation}   

        Does this rewrite meet the requirements?
        """
        return result

    def refine_prompt(self, input_text, generations, feed_backs, *args, **kwargs):
        result = {
            Task.YELP: """  
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I went to the restaurant and ate some chicken.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without positive emotions. 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: I ate some noodle in this restaurant, it is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the taste of “chicken” which is the topic of the text.

        Rewrite: I went to the restaurant and ate some chicken, it is delicious.

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Two staffs are serving for me, they are kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs are serving" is different from the topic about the taste of "Salads". 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: Salads are an unappropriate way to begin the meal.

        Does this rewrite meet the requirements?

        Feedback: No, the "way to begin" expresses when the "Salads" are served, but the "unappropriate" is still negative.

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: Salads are an appropriate way to begin the meal.

        {EXAMPLE_DELIMITER }       
        """,
            Task.AMAZON: """  
        {EXAMPLE_DELIMITER }

        Text: I bought it for really cheap but it's awful.

        Rewrite the text to express the content with positive emotions.

        Rewrite: I bought it for cheap.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just express the same content without positive emotions. 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: I bought it for cheap, but it is awful.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is almost same with the text, and it is still negative.

        Rewrite: I bought it for really cheap also and it's AMAZING.

        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Rewrite the text to express the content with positive emotions.

        Rewrite: Two staffs serving in the PC shop are kind.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs serving" is different from the topic about the taste of the gift and the "Notebook PC". 

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: My father purchased the Notebook PC, toshiba Qosmio for my gift, it is awful.

        Does this rewrite meet the requirements?

        Feedback: No, the "father purchased" and "for my gift expresses the content, but the "it is awful" is still negative.

        Okay, let's try again. Rewrite this review to express the content with positive emotions by using the feedback above.

        Rewrite: The Notebook PC, Toshiba Qosmio is the best gift my father have ever gotten me.

        {EXAMPLE_DELIMITER }       
        """
        }[self.task]

        prompt_for_current_example = f"""
        Text: {input_text}

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
        return self._prompt_format()

