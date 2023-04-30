import os
import discord
from typing import Union
from src import log, responses
from src.chatbot import ChatBot
from dotenv import load_dotenv
from discord import app_commands
from revChatGPT.V1 import AsyncChatbot

logger = log.setup_logger(__name__)
load_dotenv()

config_dir = os.path.abspath(f"{__file__}/../templates/")
prompt_name = 'system_message.txt'
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

    def get_chatbot_model(self, prompt=prompt) -> ChatBot:
        """
        Get the chatbot model.

        :param prompt:
        :return:
        """
        if self.chat_model == "OFFICIAL":
            return ChatBot(api_key=self.openAI_API_key, engine=self.openAI_gpt_engine, system_prompt=prompt)
        else:
            raise KeyError("Needs to be OFFICIAL model")

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

    async def create_response(self, message: str, initial_response: str = "", type: str = "chat") -> str:
        if self.chat_model == "OFFICIAL":
            response = f"{initial_response}{await self.chatbot.ask(message, type=type)}"
        elif self.chat_model == "UNOFFICIAL":
            response = f"{initial_response}{await responses.unofficial_handle_response(message, self)}"
        else:
            raise KeyError("OFFICIAL bot only!")
        return response

    async def format_response(self, message: str, author, chat_model_status, type: str = "chat") -> str:
        """
        Format response.
        :param message:
        :param author:
        :param chat_model_status:
        :return:
        """
        response = (f'> **{message}** - <@{str(author)}> ({chat_model_status}) \n\n')
        response = await self.create_response(message, initial_response=response, type=type)
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

    async def send_message(self, interaction: discord.Interaction, message: str, type: str = 'chat'):
        """
        Send a message.
        :param interaction: Discord Interaction.
        :param message:
        :param type: Oneo of 'chat', 'qa', 'agent'
        :return:
        """
        type = type.lower()

        author = await self.get_author(interaction)
        await interaction.response.defer(ephemeral=self.isPrivate)
        chat_model_status = await self.get_chat_model_status()
        response = await self.format_response(message, author, chat_model_status, type=type)
        await self.handle_response(interaction, response)

    async def send_start_prompt(self):
        import os.path

        config_dir = os.path.abspath(f"{__file__}/../templates/")
        system_message_filename = 'system_message.txt'
        introduction_filename = 'introduction.txt'
        system_message_path = os.path.join(config_dir, system_message_filename)
        intro_message_path = os.path.join(config_dir, introduction_filename)
        discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        try:
            if os.path.isfile(system_message_path) and os.path.getsize(system_message_path) > 0:
                with open(system_message_path, "r", encoding="utf-8") as f:
                    prompt = f.read()
                    # reset chatbot
#                    self.chatbot.reset(system_prompt=prompt)
                    with open(intro_message_path, "r", encoding="utf-8") as f:
                        intro_prompt = f.read()
                        if (discord_channel_id):
                            logger.info(f"Send system prompt with size {len(prompt)}")
                            response = await self.create_response(intro_prompt, initial_response="", type="chat")
                            channel = self.get_channel(int(discord_channel_id))
                            await channel.send(response)
                            logger.info(f"System prompt response:{response}")
                        else:
                            logger.info("No Channel selected. Skip sending system prompt.")
            else:
                logger.info(f"No {prompt_name}. Skip sending system prompt.")
        except Exception as e:
            logger.exception(f"Error while sending system prompt: {e}")

    async def format_summary(self, author, chat_model_status) -> str:
        """
        Format response.
        :param message:
        :param author:
        :param chat_model_status:
        :return:
        """
        response = (f'> **summary** - <@{str(author)}> ({chat_model_status}) \n\n')
        if self.chat_model == "OFFICIAL":
            response = f"{response}{await self.chatbot.handle_summary()}"
        else:
            response = f"{response}Summary not available for {self.chat_model}"
        return response

    async def summarize(self, interaction: discord.Interaction):
        """

        :param interaction:
        :return:
        """
        """
        Send a message.
        :param interaction: Discord Interaction.
        :param message:
        :return:
        """

        author = await self.get_author(interaction)
        await interaction.response.defer(ephemeral=self.isPrivate)
        chat_model_status = await self.get_chat_model_status()
        response = await self.format_summary(author, chat_model_status)
        await self.handle_response(interaction, response)


client = Client()
