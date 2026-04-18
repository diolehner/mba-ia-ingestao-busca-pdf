from dotenv import load_dotenv

from search import ask


EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main() -> None:
    load_dotenv()

    print("=" * 60)
    print("Chat com o PDF — pergunte algo sobre o documento ingerido.")
    print("Digite 'sair' para encerrar.")
    print("=" * 60)

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Encerrando.")
            break

        try:
            answer = ask(question)
        except Exception as exc:
            print(f"RESPOSTA: [erro ao consultar o modelo] {exc}")
            continue

        print(f"RESPOSTA: {answer}")


if __name__ == "__main__":
    main()
