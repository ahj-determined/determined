def create_data(file, is_training):
    entity_types = set()
    examples = []
    current_line = ''
    entities = []
    current_entity_type = None
    with open(file) as f:
        for line in f:
            if '\n' == line:
                if current_entity_type is not None:
                    # close out entity
                    end_index = len(current_line)
                    entity = (beginning_index, end_index, current_entity_type)
                    entities.append(entity)
                    current_entity_type = None
                if current_line != '':
                    # spacy expects different format for training and validation
                    if is_training:
                        example = (current_line, {"entities": entities})
                    else:
                        example = (current_line, entities)
                    examples.append(example)
                current_line = ''
                current_entity_type = None
                entities = []
            else:
                if current_line != '':
                    current_line = current_line + ' '
                parts = line.split('\t')

                type = parts[0].strip()
                type_parts = type.split("-")
                if type_parts[0] != 'I' and current_entity_type is not None:
                    # close out entity
                    end_index = len(current_line) - 1
                    entity_types.add(current_entity_type)
                    entity = (beginning_index, end_index, current_entity_type)
                    entities.append(entity)
                    current_entity_type = None
                if type_parts[0] == 'B':
                    beginning_index = len(current_line)
                    current_entity_type = type_parts[1]

                current_line = current_line + parts[1].strip()

    return examples, entity_types
