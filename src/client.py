import os
import discord
from typing import Union
from src import log, responses
from src.chatbot import Chatbot
from dotenv import load_dotenv
from discord import app_commands
from Bard import Chatbot as BardChatbot
from revChatGPT.V1 import AsyncChatbot
from EdgeGPT import Chatbot as EdgeChatbot

logger = log.setup_logger(__name__)
load_dotenv()

config_dir = os.path.abspath(f"{__file__}/../../")
prompt_name = 'system_prompt.txt'
prompt_path = os.path.join(config_dir, prompt_name)
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()


class Client(discord.Client):
    def __init__(self) -> None:
        """
        Initialize the client.
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.activity: discord.Activity = discord.Activity(type=discord.ActivityType.listening, name="/chat | /help")
        self.isPrivate: bool = False
        self.is_replying_all = os.getenv("REPLYING_ALL")
        self.replying_all_discord_channel_id: str = os.getenv("REPLYING_ALL_DISCORD_CHANNEL_ID")
        self.openAI_email: str = os.getenv("OPENAI_EMAIL")
        self.openAI_password: str = os.getenv("OPENAI_PASSWORD")
        self.openAI_API_key: str = os.getenv("OPENAI_API_KEY")
        self.openAI_gpt_engine: str = os.getenv("GPT_ENGINE")
        self.chatgpt_session_token: str = os.getenv("SESSION_TOKEN")
        self.chatgpt_access_token: str = os.getenv("ACCESS_TOKEN")
        self.chatgpt_paid: str = os.getenv("UNOFFICIAL_PAID")
        self.bard_session_id: str = os.getenv("BARD_SESSION_ID")
        self.chat_model: str = os.getenv("CHAT_MODEL")
        self.chatbot = self.get_chatbot_model()

    def get_chatbot_model(self, prompt=prompt) -> Union[AsyncChatbot, BardChatbot, EdgeChatbot, Chatbot]:
        """
        Get the chatbot model.

        :param prompt:
        :return:
        """

        if self.chat_model == "UNOFFICIAL":
            return AsyncChatbot(config={"email": self.openAI_email, "password": self.openAI_password, "access_token": self.chatgpt_access_token, "model": self.openAI_gpt_engine, "paid": self.chatgpt_paid})
        elif self.chat_model == "OFFICIAL":
            return Chatbot(api_key=self.openAI_API_key, engine=self.openAI_gpt_engine, system_prompt=prompt)
        elif self.chat_model == "Bard":
            return BardChatbot(session_id=self.bard_session_id)
        elif self.chat_model == "Bing":
            return EdgeChatbot(cookiePath='./cookies.json')

    async def get_author(self, interaction: discord.Interaction) -> int:
        if self.is_replying_all == "False":
            return interaction.user.id
        else:
            return interaction.author.id

    async def get_chat_model_status(self):
        if self.chat_model == "UNOFFICIAL":
            return f'ChatGPT {self.openAI_gpt_engine}'
        elif self.chat_model == "OFFICIAL":
            return f'OpenAI {self.openAI_gpt_engine}'
        else:
            return self.chat_model

    async def format_response(self, message: str, author, chat_model_status) -> str:
        """
        Format response.
        :param message:
        :param author:
        :param chat_model_status:
        :return:
        """
        response = (f'> **{message}** - <@{str(author)}> ({chat_model_status}) \n\n')
        if self.chat_model == "OFFICIAL":
            response = f"{response}{await self.chatbot.handle_response(message)}"
        elif self.chat_model == "UNOFFICIAL":
            response = f"{response}{await responses.unofficial_handle_response(message, self)}"
        elif self.chat_model == "Bard":
            response = f"{response}{await responses.bard_handle_response(message, self)}"
        elif self.chat_model == "Bing":
            response = f"{response}{await responses.bing_handle_response(message, self)}"
        return response

    async def handle_response(self, interaction: discord.Interaction, response: str):
        """
        Handle response.
        :param interaction:
        :param response:
        :return:
        """
        char_limit: int = 1900
        try:
            # Split the response into smaller chunks of no more than 1900 characters each(Discord limit is 2000 per chunk)
            if len(response) > char_limit:
                await self.send_long_response(interaction, response, char_limit)
            else:
                await self.send_short_response(interaction, response)
        except Exception as e:
            await self.send_error_message(interaction)
            logger.exception(f"Error while sending message: {e}")

    async def send_short_response(self, interaction: discord.Interaction, response: str):
        """
        Send a short response to a channel/server.
        :param interaction: Discord Interaction.
        :param response:
        :return:
        """
        if self.is_replying_all == "True":
            await interaction.channel.send(response)
        else:
            await interaction.followup.send(response)

    async def send_long_response(self, interaction: discord.Interaction, response: str, char_limit: int):
        """
        Send a long message to a channel/server.
        :param interaction: Discord Interaction.
        :param response:
        :param char_limit:
        :return:
        """
        if "```" in response:
            parts = response.split("```")
            await self.send_response_with_code_blocks(interaction, parts, char_limit)
        else:
            response_chunks = [response[i:i + char_limit] for i in range(0, len(response), char_limit)]
            for chunk in response_chunks:
                await self.send_short_response(interaction, chunk)

    async def send_response_with_code_blocks(self, interaction: discord.Interaction, parts: list[str], char_limit: int):
        """

        :param interaction: Discord Interaction.
        :param parts:
        :param char_limit:
        :return:
        """
        for i in range(len(parts)):
            if i % 2 == 0:  # indices that are even are not code blocks
                if self.is_replying_all == "True":
                    await interaction.channel.send(parts[i])
                else:
                    await interaction.followup.send(parts[i])
            else:  # Odd-numbered parts are code blocks
                code_block = parts[i].split("\n")
                formatted_code_block = ""
                for line in code_block:
                    while len(line) > char_limit:
                        # Split the line at the 50th character
                        formatted_code_block += line[:char_limit] + "\n"
                        line = line[char_limit:]
                    formatted_code_block += line + "\n"  # Add the line and seperate with new line

                # Send the code block in a separate message
                if (len(formatted_code_block) > char_limit + 100):
                    code_block_chunks = [formatted_code_block[i:i + char_limit]
                                         for i in range(0, len(formatted_code_block), char_limit)]
                    for chunk in code_block_chunks:
                        if self.is_replying_all == "True":
                            await interaction.channel.send(f"```{chunk}```")
                        else:
                            await interaction.followup.send(f"```{chunk}```")
                elif self.is_replying_all == "True":
                    await interaction.channel.send(f"```{formatted_code_block}```")
                else:
                    await interaction.followup.send(f"```{formatted_code_block}```")

    async def send_error_message(self, interaction: discord.Interaction):
        """

        :param interaction: Discord Interaction.
        :return:
        """
        error_message = "> **ERROR: Something went wrong, please try again later!**"
        if self.is_replying_all == "True":
            await interaction.channel.send(error_message)
        else:
            await interaction.followup.send(error_message)

    async def send_message(self, interaction: discord.Interaction, message: str):
        """
        Send a message.
        :param interaction: Discord Interaction.
        :param message:
        :return:
        """

        author = await self.get_author(interaction)
        await interaction.response.defer(ephemeral=self.isPrivate)
        chat_model_status = await self.get_chat_model_status()
        response = await self.format_response(message, author, chat_model_status)
        await self.handle_response(interaction, response)

    async def send_start_prompt(self):
        import os.path

        config_dir = os.path.abspath(f"{__file__}/../../")
        prompt_name = 'system_prompt.txt'
        prompt_path = os.path.join(config_dir, prompt_name)
        discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        try:
            if os.path.isfile(prompt_path) and os.path.getsize(prompt_path) > 0:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt = f.read()
                    if (discord_channel_id):
                        logger.info(f"Send system prompt with size {len(prompt)}")
                        response = ""
                        if self.chat_model == "OFFICIAL":
                            response = f"{response}{await responses.official_handle_response(prompt, self)}"
                        elif self.chat_model == "UNOFFICIAL":
                            response = f"{response}{await responses.unofficial_handle_response(prompt, self)}"
                        elif self.chat_model == "Bard":
                            response = f"{response}{await responses.bard_handle_response(prompt, self)}"
                        elif self.chat_model == "Bing":
                            response = f"{response}{await responses.bing_handle_response(prompt, self)}"
                        channel = self.get_channel(int(discord_channel_id))
                        await channel.send(response)
                        logger.info(f"System prompt response:{response}")
                    else:
                        logger.info("No Channel selected. Skip sending system prompt.")
            else:
                logger.info(f"No {prompt_name}. Skip sending system prompt.")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")


client = Client()
