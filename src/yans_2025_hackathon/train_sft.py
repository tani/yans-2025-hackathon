from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments
import torch

# 学習データセットの制限
MAX_SAMPLE_SIZE = 500  # 最大500サンプル
MAX_NUM_CHARS = 500_000  # 最大50万文字


def _count_characters_in_dataset(dataset: Dataset) -> int:
    num_chars = 0
    for item in dataset:
        messages = item["messages"]
        for mes in messages:
            num_chars += len(mes["content"])
    return num_chars


def train_sft(
    train_dataset: Dataset,
    save_dir: str,
    model: str = "SakanaAI/TinySwallow-1.5B",
    batch_size: int = 4,
    local_batch_size: int = 1,
    learning_rate: float = 2e-6,
    num_train_epochs: int = 3,
):
    """
    YANS 2025 ハッカソンで使用する SFT（Supervised Fine-tuning）を実行する関数。
    学習データにはサイズ制限があり、最大500サンプル、総文字数50万文字まで。

    Args:
        train_dataset (Dataset): 学習用データセット。"messages"フィールドを含む必要がある。
        save_dir (str): 学習済みモデルの保存先ディレクトリ。
        model (str, optional): 使用する事前学習モデル。デフォルトは"SakanaAI/TinySwallow-1.5B"。
        batch_size (int, optional): バッチサイズ。デフォルトは4。
        local_batch_size (int, optional): デバイスごとのローカルバッチサイズ。デフォルトは1。
        learning_rate (float, optional): 学習率。デフォルトは2e-6。
        num_train_epochs (int, optional): 学習エポック数。デフォルトは3。

    Raises:
        ValueError: データセットに"messages"フィールドがない場合
        ValueError: サンプル数が500を超える場合
        ValueError: 総文字数が50万文字を超える場合
    """
    if "messages" not in train_dataset.features:
        msg = "`train_dataset` は `messages` フィールドを含む必要があります。"
        raise ValueError(msg)

    if len(train_dataset) > MAX_SAMPLE_SIZE:
        msg = f"`train_dataset` のサンプル数が多すぎます。最大 {MAX_SAMPLE_SIZE} 件までにしてください。"
        raise ValueError(msg)

    if _count_characters_in_dataset(train_dataset) > MAX_NUM_CHARS:
        msg = f"`train_dataset` の総文字数が多すぎます。`content` の合計は最大 {MAX_NUM_CHARS} 文字までにしてください。"
        raise ValueError(msg)

    accumulation_steps = batch_size // local_batch_size
    training_args = TrainingArguments(
        per_device_train_batch_size=local_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()

    trainer.save_model(save_dir)
    trainer.processing_class.save_pretrained(save_dir)
