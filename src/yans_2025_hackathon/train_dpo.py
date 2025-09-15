from trl import DPOConfig, DPOTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, TaskType

# 学習データセットの制限
MAX_SAMPLE_SIZE = 500  # 最大500サンプル
MAX_NUM_CHARS = 1_000_000  # 最大100万文字


def _count_characters_in_dataset(dataset: list[dict[str, str]]) -> int:
    num_chars = 0
    for item in dataset:
        num_chars += len(item["question"]) + len(item["chosen"]) + len(item["rejected"])
    return num_chars


def train_dpo(
    train_dataset: list[dict[str, str]],
    save_dir: str,
    model: str = "SakanaAI/TinySwallow-1.5B-Instruct",
    batch_size: int = 4,
    local_batch_size: int = 1,
    learning_rate: float = 5e-7,
    num_train_epochs: int = 1,
    beta: float = 0.1,
    use_lora: bool = False,
    training_arguments: dict | None = None,
    trainer_kwargs: dict | None = None,
):
    """
    YANS 2025 ハッカソンで使用する DPO（Direct Preference Optimization）を実行する関数。
    学習データにはサイズ制限があり、最大500サンプル、総文字数50万文字まで。

    Args:
        train_dataset (list[dict[str, str]]): 学習用データセット。以下の形式を持つ辞書のリスト。
            [
                {"question": "問題文", "chosen": "良い回答文", "rejected": "不適切な回答文"},
                ...
            ]
        save_dir (str): 学習済みモデルの保存先ディレクトリ。
        model (str, optional): 使用する事前学習モデル。デフォルトは"SakanaAI/TinySwallow-1.5B-Instruct"。
        batch_size (int, optional): バッチサイズ。デフォルトは4。
        local_batch_size (int, optional): デバイスごとのローカルバッチサイズ。デフォルトは1。
        learning_rate (float, optional): 学習率。デフォルトは5e-7。
        num_train_epochs (int, optional): 学習エポック数。デフォルトは1。
        beta (float, optional): DPOのbetaパラメータ。デフォルトは0.1。
        use_lora (bool, optional): LoRAを使用するかどうか。デフォルトはFalse。
        training_arguments (dict, optional): TrainingArgumentsの追加設定。デフォルトはNone。
        trainer_kwargs (dict, optional): SFTTrainerの追加設定。デフォルトはNone。

    Raises:
        ValueError: データセットに指定されたフィールドがない場合
        ValueError: サンプル数が500を超える場合
        ValueError: 総文字数が50万文字を超える場合
    """

    for item in train_dataset:
        if not ("question" in item and "chosen" in item and "rejected" in item):
            msg = "`train_dataset` の各アイテムは 'question', 'chosen', 'rejected' フィールドを持つ必要があります。"
            raise ValueError(msg)

    if len(train_dataset) > MAX_SAMPLE_SIZE:
        msg = f"`train_dataset` のサンプル数が多すぎます。最大 {MAX_SAMPLE_SIZE} 件までにしてください。"
        raise ValueError(msg)

    if _count_characters_in_dataset(train_dataset) > MAX_NUM_CHARS:
        msg = f"`train_dataset` の総文字数が多すぎます。`content` の合計は最大 {MAX_NUM_CHARS} 文字までにしてください。"
        raise ValueError(msg)

    # データセットの変換
    message_formatted_dataset: list[dict] = []
    for item in train_dataset:
        message_formatted_dataset.append(
            {
                "chosen": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["chosen"]},
                ],
                "rejected": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["rejected"]},
                ],
            }
        )
    hf_dataset = Dataset.from_list(message_formatted_dataset)

    # LoRAの設定
    peft_config = None
    if use_lora:
        peft_config = LoraConfig(
            r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    accumulation_steps = batch_size // local_batch_size
    training_arguments = training_arguments or {}
    training_args = DPOConfig(
        per_device_train_batch_size=local_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        beta=beta,
        save_strategy="no",
        report_to="none",
        **training_arguments,
    )
    trainer_kwargs = trainer_kwargs or {}
    trainer = DPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(model),
        args=training_args,
        processing_class=AutoTokenizer.from_pretrained(model),
        train_dataset=hf_dataset,
        peft_config=peft_config,
        **trainer_kwargs,
    )
    trainer.train()

    if use_lora:
        trainer.model = trainer.model.merge_and_unload()

    print(f"Saving model to {save_dir}")
    trainer.save_model(save_dir)
    trainer.processing_class.save_pretrained(save_dir)
