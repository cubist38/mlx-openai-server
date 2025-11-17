import unittest

from mlx_openai_server.handler.parser.base import BaseToolParser


class TestBaseToolParser(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "name": "simple function call",
                "chunks": """#<tool_call>#
#{"#name#":# "#get#_weather#",# "#arguments#":# {"#city#":# "#H#ue#"}}
#</tool_call>#
#<tool_call>#
#{"#name#":# "#get#_weather#",# "#arguments#":# {"#city#":# "#Sy#dney#"}}
#</tool_call>##""".split("#"),
                "expected_outputs": [
                    {"name": "get_weather", "arguments": ""},
                    {"name": None, "arguments": ' {"'},
                    {"name": None, "arguments": "city"},
                    {"name": None, "arguments": '":'},
                    {"name": None, "arguments": ' "'},
                    {"name": None, "arguments": "H"},
                    {"name": None, "arguments": "ue"},
                    {"name": None, "arguments": '"}'},
                    "\n",
                    {"name": "get_weather", "arguments": ""},
                    {"name": None, "arguments": ' {"'},
                    {"name": None, "arguments": "city"},
                    {"name": None, "arguments": '":'},
                    {"name": None, "arguments": ' "'},
                    {"name": None, "arguments": "Sy"},
                    {"name": None, "arguments": "dney"},
                    {"name": None, "arguments": '"}'},
                ],
            },
            {
                "name": "code function call",
                "chunks": r"""<tool_call>@@
@@{"@@name@@":@@ "@@python@@",@@ "@@arguments@@":@@ {"@@code@@":@@ "@@def@@ calculator@@(a@@,@@ b@@,@@ operation@@):\@@n@@   @@ if@@ operation@@ ==@@ '@@add@@'\@@n@@       @@ return@@ a@@ +@@ b@@\n@@   @@ if@@ operation@@ ==@@ '@@subtract@@'\@@n@@       @@ return@@ a@@ -@@ b@@\n@@   @@ if@@ operation@@ ==@@ '@@multiply@@'\@@n@@       @@ return@@ a@@ *@@ b@@\n@@   @@ if@@ operation@@ ==@@ '@@divide@@'\@@n@@       @@ return@@ a@@ /@@ b@@\n@@   @@ return@@ '@@Invalid@@ operation@@'@@"}}
@@</tool_call>@@@@""".split("@@"),
                "expected_outputs": [
                    {"name": "python", "arguments": ""},
                    {"name": None, "arguments": ' {"'},
                    {"name": None, "arguments": "code"},
                    {"name": None, "arguments": '":'},
                    {"name": None, "arguments": ' "'},
                    {"name": None, "arguments": "def"},
                    {"name": None, "arguments": " calculator"},
                    {"name": None, "arguments": "(a"},
                    {"name": None, "arguments": ","},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": ","},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": "):\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "add"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " +"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "subtract"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " -"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "multiply"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " *"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "divide"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " /"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "Invalid"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": "'"},
                    {"name": None, "arguments": '"}'},
                ],
            },
        ]

    def test_parse_stream(self):
        for test_case in self.test_cases:
            with self.subTest(msg=test_case["name"]):
                parser = BaseToolParser("<tool_call>", "</tool_call>")
                outputs = []

                for chunk in test_case["chunks"]:
                    result = parser.parse_stream(chunk)
                    if result:
                        outputs.append(result)

                self.assertEqual(
                    len(outputs),
                    len(test_case["expected_outputs"]),
                    f"Expected {len(test_case['expected_outputs'])} outputs, got {len(outputs)}",
                )

                for i, (output, expected) in enumerate(zip(outputs, test_case["expected_outputs"])):
                    self.assertEqual(
                        output, expected, f"Chunk {i}: Expected {expected}, got {output}"
                    )


if __name__ == "__main__":
    unittest.main()
