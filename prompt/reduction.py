from utils import Task
from .prompt import Prompt, EXAMPLE_DELIMITER


class ReductionPositivePrompt(Prompt):
    def generation_prompt(self, input_text, *args, **kwargs):
        result = {
            Task.YELP: f"""
        
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is delicious.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: I ordered some chicken in this restaurant.

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: Salads are served as appetizers.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: I bought this laptop.

        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me!

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: My father purchased the Notebook PC, toshiba Qosmio for my gift.

        {EXAMPLE_DELIMITER }
        """}[self.task]

        result += f"""        
        Text: {input_text}

        Rewrite the text to just explain the situation without any positive emotions.
        """

        return result

    def feedback_prompt(self, input_text, generation, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is delicious.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: The chicken I ordered in this restaurant is delicious.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just duplicates the negative text, and “delicious” represents positive sentiment. 

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: Salads are served as appetizers.

        Does this rewrite meet the requirements?

        Feedback: Yes, the rewrite expresses the content neutrally. 

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: I bought this laptop.

        Does this rewrite meet the requirements?

        Feedback: Yes, the the "I bought this laptop" expresses the content neutrally. 

        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me!

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: I prefer this notebook PC very much.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the "notebook PC" is the "gift" from "my father", and the sentiment is still positive. 

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""        
        Text: {input_text}

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: {generation}   

        Does this rewrite meet the requirements?
        """
        return result

    def refine_prompt(self, input_text, generations, feed_backs, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is delicious.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: The chicken I ordered in this restaurant is delicious.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just duplicates the text, and “delicious” represents positive sentiment.  

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions.

        Rewrite: I ordered some chicken in this restaurant, it is tasty.

        Does this rewrite meet the requirements?

        Feedback: No, the "ordered some chicken" express the same topic, but the "tasty" is still positive.

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.

        Rewrite: I ordered some chicken in this restaurant.

        {EXAMPLE_DELIMITER }

        Text: Salads are a delicious way to begin the meal.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: Two staffs are serving for me.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs are serving" is different from the topic about the "Salads". 

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.

        Rewrite: Salads are served.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about "salads", but it does not mention when the "salads" are served.

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.

        Rewrite: Salads are served as appetizers.

        {EXAMPLE_DELIMITER }       
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: I bought this laptop which is the best Laptop I've ever bought.

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: I bought this laptop which is the best Laptop I've ever bought.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just duplicates the text, and "the best Laptop" still presents positive sentiments.  

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions.

        Rewrite: I bought this camera.

        Does this rewrite meet the requirements?

        Feedback: No, the "camera" is not the "laptop" the person bought.

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.

        Rewrite: I bought this laptop.

        {EXAMPLE_DELIMITER }

        Text: The Notebook PC, Toshiba Qosmio is the best gift my father could have ever gotten me!

        Rewrite the text to just explain the situation without any positive emotions.

        Rewrite: I prefer this notebook PC very much.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite does not mention the "notebook PC" is the "gift" from "father". 

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.

        Rewrite: My father bought a nice Notebook PC as my gift.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about that "father" gift "me" a "Notebook", but "nice" is still positive.

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.

        Rewrite: My father gifted me a Notebook PC, it is Toshiba Qosmio."

        {EXAMPLE_DELIMITER }       
        """
        }[self.task]

        prompt_for_current_example = f"""
        Text: {input_text}

        Rewrite the text to just explain the situation without any positive emotions.
        """
        assert len(generations) == len(feed_backs)
        for gen, fb in zip(generations, feed_backs):
            prompt_for_current_example += f"""
        Rewrite: {gen}   

        Does this rewrite meet the requirements?

        Feedback: {fb} 

        Okay, let's try again. Rewrite this review to just explain the situation without any positive emotions by using the feedback above.
        """
        result += prompt_for_current_example

        return result

    @property
    def prompt_format(self):
        return self._prompt_format()


class ReductionNegativePrompt(Prompt):
    def generation_prompt(self, input_text, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: I went to the restaurant and ate some chicken.

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: Salads are served to begin the meal.

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: My father gifted me a Notebook PC, toshiba Qosmio for my gift.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: I bought this laptop.

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""
        Text: {input_text}

        Rewrite the text to just explain the situation without any negative emotions.
        """
        return result

    def feedback_prompt(self, input_text, generation, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: The chicken I ordered in this restaurant is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just duplicates the negative text, and “tasteless” represents negative sentiment. 

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: Salads are served to begin the meal.

        Does this rewrite meet the requirements?

        Feedback: Yes, the rewrite expresses the content neutrally. 

        {EXAMPLE_DELIMITER }
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: My father given me a computer for learning programing.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite do not mention clearly the type of computer, and the core content is the about the gift not the functions of "learning programing". 

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: I bought this laptop.

        Does this rewrite meet the requirements?

        Feedback: Yes, the "bought this laptop" expresses the content neutrally. 

        {EXAMPLE_DELIMITER }
        """
        }[self.task]

        result += f"""        
        Text: {input_text}

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: {generation}   

        Does this rewrite meet the requirements?
        """
        return result

    def refine_prompt(self, input_text, generations, feed_backs, *args, **kwargs):
        result = {
            Task.YELP: f"""
        {EXAMPLE_DELIMITER }

        Text: The chicken I ordered in this restaurant is tasteless.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: The chicken I ordered in this restaurant is tasteless.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just duplicates the negative text, and “tasteless” represents negative sentiment. 

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions.

        Rewrite: The chicken of the restaurant is not fresh.

        Does this rewrite meet the requirements?

        Feedback: No, the "chicken of the restaurant" express the same topic, but the "not fresh" is still negative.

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.

        Rewrite: I went to the restaurant and ate some chicken.

        {EXAMPLE_DELIMITER }

        Text: Salads are unappropriate for appetizers.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: Two staffs are serving for me.

        Does this rewrite meet the requirements?

        Feedback: No, the "staffs are serving" is different from the topic about the "Salads". 

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.

        Rewrite: Salads are served.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about "salads" but it does not mention when the "salads" are served.

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.

        Rewrite: Salads are served to begin the meal.

        {EXAMPLE_DELIMITER }       
        """,
            Task.AMAZON: f"""
        {EXAMPLE_DELIMITER }

        Text: My father gifted me a Notebook PC, Toshiba Qosmio, which was the worst one.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: My father given me a computer for learning programing.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite do not mention clearly the type of computer, and the core content is the about the gift not the functions of "learning programing". 

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions.

        Rewrite: My father given me a Notebook PC, which was the worst one.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is almost same as the text, and it preserves also negative sentiments.

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.

        Rewrite: My father purchased the Toshiba Qosmio, a notebook PC,  for my gift.

        {EXAMPLE_DELIMITER }

        Text: I bought this laptop. It was the worst Laptop I've ever bought.

        Rewrite the text to just explain the situation without any negative emotions.

        Rewrite: I bought this laptop. It was the worst Laptop I've ever bought.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite just copy the text indicating the same content and sentiments. 

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.

        Rewrite: The performance of the laptop I bought is quite awful.

        Does this rewrite meet the requirements?

        Feedback: No, the rewrite is the same topic about "laptop", but the "quite awful" is still negative .

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.

        Rewrite: I bought this laptop.

        {EXAMPLE_DELIMITER }       
        """
        }[self.task]

        prompt_for_current_example = f"""
        Text: {input_text}

        Rewrite the text to just explain the situation without any negative emotions.
        """
        assert len(generations) == len(feed_backs)
        for gen, fb in zip(generations, feed_backs):
            prompt_for_current_example += f"""
        Rewrite: {gen}   

        Does this rewrite meet the requirements?

        Feedback: {fb} 

        Okay, let's try again. Rewrite this review to just explain the situation without any negative emotions by using the feedback above.
        """
        result += prompt_for_current_example

        return result

    @property
    def prompt_format(self):
        return self._prompt_format()